#
# Copyright (C) 2026  FreeIPA Contributors see COPYING for license
#

"""
ML-DSA (post-quantum) certificate integration tests.

Reuses :class:`~ipatests.test_integration.test_cert.TestInstallMasterClient`
for legacy certmonger/profile scenarios. All PQC install options, helpers,
and ML-DSA-specific test classes live in this module; ``test_cert.py`` is
unchanged for RSA/default reference. Explicit ML-DSA rekey/resubmit/expiry
coverage lives on :class:`TestInstallMasterClientMLDSACA` (helpers mirror
:class:`~ipatests.test_integration.test_cert.TestCertmongerRekey`).
"""

# pylint: disable=no-member
# Multiple inheritance pattern used throughout this module:
# Test classes inherit from both helper mixins (PQCInstallBase, PQCCertHelpers)
# and IntegrationTest. The IntegrationTest base class provides master, clients,
# replicas attributes via pytest fixtures, which pylint cannot detect.

import os
import random
import string
import time

import pytest

from ipaplatform.paths import paths
from ipapython.dn import DN

from ipatests.pytest_ipa.integration import tasks
from ipatests.test_integration.base import IntegrationTest
from ipatests.test_integration.test_cert import TestInstallMasterClient


def _expected_ml_dsa_httpd_public_key_label(ipa_key_type):
    """Expected substring on openssl x509 Public-Key output for httpd."""
    if not ipa_key_type:
        return None
    kt = ipa_key_type.strip().lower()
    if not kt.startswith('mldsa'):
        return None
    if kt == 'mldsa':
        return 'ML-DSA-65'
    if ':' in ipa_key_type:
        strength = ipa_key_type.split(':', 1)[1].strip()
        if strength.isdigit():
            return 'ML-DSA-{}'.format(strength)
    return None


def _mldsa_openssl_algorithm(ipa_key_type, mldsa_cert_keygen=None):
    """OpenSSL ``genpkey -algorithm`` name for ML-DSA CSRs."""
    if mldsa_cert_keygen:
        return mldsa_cert_keygen
    label = _expected_ml_dsa_httpd_public_key_label(ipa_key_type)
    return label or 'ML-DSA-65'


class PQCInstallBase:
    """Base class for ML-DSA FreeIPA installations.

    Provides install() method with --key-type-size and --ca-key-type parameters
    for master+client (line topology). Subclasses inherit from this plus
    TestInstallMasterClient to get existing cert test coverage.

    Attributes:
        ipa_key_type: ML-DSA key size for IPA service certs
            (e.g., 'mldsa:65')
        ca_key_type: ML-DSA key size for CA signing keys
            (e.g., 'mldsa:87')
        mldsa_cert_keygen: OpenSSL algorithm name for certmonger
            (e.g., 'ML-DSA-65')

    Inherited from IntegrationTest (via multiple inheritance):
        master, clients, replicas: Test host instances
        domain_level, token_password, random_serial: Configuration attributes
    """

    ipa_key_type = None
    ca_key_type = None
    mldsa_cert_keygen = None

    @classmethod
    def _install_line_master_client_with_pqc(cls, mh):
        extra_args = []
        if cls.domain_level is not None:
            domain_level = cls.domain_level
        else:
            domain_level = cls.master.config.domain_level

        if cls.token_password:
            extra_args.extend(('--token-password', cls.token_password,))
        if cls.ipa_key_type:
            extra_args.extend(['--key-type-size', cls.ipa_key_type])
        if cls.ca_key_type:
            extra_args.extend(['--ca-key-type', cls.ca_key_type])

        tasks.install_master(
            cls.master,
            setup_dns=True,
            domain_level=domain_level,
            random_serial=cls.random_serial,
            extra_args=extra_args,
        )
        tasks.add_a_records_for_hosts_in_master_domain(cls.master)
        tasks.install_clients([cls.master], cls.clients)

    @classmethod
    def install(cls, mh):
        cls._install_line_master_client_with_pqc(mh)
        result = cls.clients[0].run_command(['date', '+%Y-%m-%d %H:%M:%S'])
        cls.since = result.stdout_text.strip()

    def test_getcert_list_profile(self):
        """Profile listing plus ML-DSA httpd public key when applicable."""
        super().test_getcert_list_profile()
        ml_label = _expected_ml_dsa_httpd_public_key_label(self.ipa_key_type)
        if ml_label:
            pk = self.master.run_command(
                'openssl x509 -text -noout -in %s | grep Public-Key'
                % paths.HTTPD_CERT_FILE
            ).stdout_text
            assert ml_label in pk


class PQCCertHelpers:
    """Reusable helper methods for ML-DSA certificate operations.

    Provides CSR generation, certmonger enrollment, rekey/resubmit, and
    cleanup helpers for both RSA and ML-DSA certificates. All helpers use
    try/finally for proper resource cleanup.

    Methods mirror the RSA patterns in
    :class:`~ipatests.test_integration.test_cert.TestCertmongerRekey` and
    resubmit flows, but assert via OpenSSL ``Public-Key`` labels and pubkey
    fingerprints (cryptography ``key_size`` does not apply to ML-DSA).
    Expiry autorenew uses :meth:`_simulate_cert_expiry` with a short-lived
    profile so the clock jump stays small on the shared topology.

    Inherited from IntegrationTest (via multiple inheritance):
        master: Master host instance for running commands
    """

    mldsa_cert_keygen = 'ML-DSA-65'
    # Baseline in current CI (0.79.21-9.fc45) lacks cross-strength ML-DSA
    # rekey; builds with certmonger PR #315 are required for 65->87.
    _certmonger_cross_rekey_broken_prefix = '0.79.21-9'

    def _mldsa_algorithm(self):
        return _mldsa_openssl_algorithm(
            self.ipa_key_type, self.mldsa_cert_keygen
        )

    def _require_openssl_mldsa(self, host=None, algorithm=None):
        host = host or self.master
        probe = os.path.join(paths.OPENSSL_PRIVATE_DIR, '.mldsa-probe.key')
        algo = algorithm or self._mldsa_algorithm()
        try:
            gen = host.run_command(
                ['openssl', 'genpkey', '-algorithm', algo, '-out', probe],
                raiseonerr=False,
            )
            if gen.returncode != 0:
                pytest.skip(
                    'OpenSSL on %s cannot generate %s keys (need OpenSSL '
                    'with ML-DSA support).' % (host.hostname, algo)
                )
        finally:
            host.run_command(['rm', '-f', probe], raiseonerr=False)

    def _generate_user_csr(self, user, stem, key_type='rsa'):
        csr_file = os.path.join(paths.OPENSSL_DIR, '%s.csr' % stem)
        key_file = os.path.join(paths.OPENSSL_PRIVATE_DIR, '%s.key' % stem)
        if key_type == 'rsa':
            self.master.run_command([
                'openssl', 'req', '-newkey', 'rsa:2048', '-keyout', key_file,
                '-nodes', '-out', csr_file, '-subj', '/CN=' + user,
            ])
        elif key_type == 'mldsa':
            self._require_openssl_mldsa(self.master)
            algo = self._mldsa_algorithm()
            self.master.run_command(
                ['openssl', 'genpkey', '-algorithm', algo, '-out', key_file]
            )
            self.master.run_command([
                'openssl', 'req', '-new', '-key', key_file, '-out', csr_file,
                '-subj', '/CN=' + user,
            ])
        else:
            raise ValueError(key_type)
        return csr_file, key_file

    def _ipa_user_cert_request(self, user, csr_file, cert_file):
        self.master.run_command([
            'ipa', 'cert-request', '--principal', user,
            '--certificate-out', cert_file, csr_file,
        ])

    def _assert_user_cert_public_key(self, cert_file, key_type):
        pk = self.master.run_command(
            'openssl x509 -in %s -noout -text | grep Public-Key' % cert_file
        ).stdout_text
        if key_type == 'rsa':
            assert 'RSA Public-Key' in pk or '2048 bit' in pk, pk
        elif key_type == 'mldsa':
            assert self._mldsa_algorithm() in pk, pk
        else:
            raise ValueError(key_type)

    def _issue_user_certs(self, user, key_specs):
        ldap = self.master.ldap_connect()
        tasks.kinit_admin(self.master)
        try:
            tasks.user_add(self.master, user)
            for stem, key_type in key_specs:
                csr_file, key_file = self._generate_user_csr(
                    user, stem, key_type
                )
                cert_file = '%s.crt' % stem
                self._ipa_user_cert_request(user, csr_file, cert_file)
                self._assert_user_cert_public_key(cert_file, key_type)
                self.master.run_command(
                    ['rm', '-f', csr_file, key_file, cert_file],
                    raiseonerr=False
                )
            entry = ldap.get_entry(
                DN(('uid', user), ('cn', 'users'), ('cn', 'accounts'),
                   self.master.domain.basedn)
            )
            assert len(entry.get('usercertificate')) == len(key_specs)
        finally:
            self.master.run_command(
                ['ipa', 'user-del', user],
                raiseonerr=False
            )
            tasks.kdestroy_all(self.master)

    def _getcert_request_host_cert(self, host, req_id, keygen=None,
                                   profile='caIPAserviceCert',
                                   auto_renew=True):
        certfile = os.path.join(paths.OPENSSL_CERTS_DIR, '%s.pem' % req_id)
        keyfile = os.path.join(paths.OPENSSL_PRIVATE_DIR, '%s.key' % req_id)
        hostname = host.hostname
        host.run_command(['rm', '-f', certfile, keyfile], raiseonerr=False)
        cmd_arg = [
            'getcert', 'request', '-c', 'ipa', '-I', req_id,
            '-k', keyfile, '-f', certfile,
            '-D', hostname, '-K', 'host/%s' % hostname,
            '-N', 'CN={}'.format(hostname),
            '-U', 'id-kp-clientAuth', '-T', profile,
        ]
        if keygen:
            # Combined form (``-G ML-DSA-65``). Avoid ``-G ML-DSA -g 65``
            # here: request expands that to ML-DSA-65 but also stores
            # key_gen_size=65, which later mismatches keyiread's real size
            # and makes resubmit regenerate the key.
            cmd_arg.extend(['-G', keygen])
        if not auto_renew:
            # Disable autorenew so expiry tests can enable it after snapshot.
            cmd_arg.append('-R')
        result = host.run_command(cmd_arg)
        assert (
            'New signing request "%s" added.\n' % req_id in result.stdout_text
        )
        status = tasks.wait_for_request(host, req_id, 300)
        assert status == 'MONITORING', (
            'certmonger request %s is in state %s' % (req_id, status)
        )
        list_out = host.run_command(
            ['getcert', 'list', '-i', req_id]
        ).stdout_text
        assert 'profile: %s' % profile in list_out
        return certfile

    def _cleanup_getcert_request(self, host, req_id):
        host.run_command(
            ['getcert', 'stop-tracking', '-i', req_id], raiseonerr=False)
        host.run_command(
            ['rm', '-f',
             os.path.join(paths.OPENSSL_CERTS_DIR, '%s.pem' % req_id),
             os.path.join(paths.OPENSSL_PRIVATE_DIR, '%s.key' % req_id)],
            raiseonerr=False,
        )

    def _cert_public_key_label(self, host, certfile):
        return host.run_command(
            'openssl x509 -in %s -noout -text | grep Public-Key' % certfile
        ).stdout_text

    def _cert_pubkey_sha256(self, host, certfile):
        """SHA-256 of the certificate's SubjectPublicKeyInfo (DER/PEM)."""
        return host.run_command(
            'openssl x509 -in %s -noout -pubkey | openssl sha256' % certfile
        ).stdout_text.strip()

    def _keyfile_sha256(self, host, req_id):
        """SHA-256 of the private key file for req_id."""
        keyfile = os.path.join(paths.OPENSSL_PRIVATE_DIR, '%s.key' % req_id)
        return host.run_command(
            ['openssl', 'sha256', keyfile]
        ).stdout_text.strip()

    def _cert_serial(self, host, certfile):
        return host.run_command(
            ['openssl', 'x509', '-in', certfile, '-noout', '-serial']
        ).stdout_text.strip()

    def _getcert_rekey(self, host, req_id, keygen=None, new_id=None):
        """Rekey like TestCertmongerRekey.

        For ML-DSA, certmonger rekey expects the full algorithm name on
        ``-G`` (``ML-DSA-44`` / ``65`` / ``87``). Bare ``-G ML-DSA -g N``
        is expanded only on *request*, not on rekey; the daemon rejects
        bare ``ML-DSA`` as KEY_TYPE. Optionally pass ``-g N`` as well to
        match certmonger's own ML-DSA rekey tests (key_gen_type + size).
        """
        cmd = ['getcert', 'rekey', '-i', req_id]
        if keygen:
            cmd.extend(['-G', keygen])
            if keygen.startswith('ML-DSA-'):
                cmd.extend(['-g', keygen.rsplit('-', 1)[-1]])
        if new_id:
            cmd.extend(['-I', new_id])
        result = host.run_command(cmd)
        wait_id = new_id or req_id
        status = tasks.wait_for_request(host, wait_id, 300)
        assert status == 'MONITORING', (
            'certmonger rekey %s is in state %s' % (wait_id, status)
        )
        return result

    def _getcert_resubmit(self, host, req_id):
        """Resubmit (renew) without generating a new key pair."""
        host.run_command(['getcert', 'resubmit', '-i', req_id])
        status = tasks.wait_for_request(host, req_id, 300)
        assert status == 'MONITORING', (
            'certmonger resubmit %s is in state %s' % (req_id, status)
        )

    def _require_certmonger_mldsa_cross_rekey(self, host=None):
        """Skip when certmonger lacks ML-DSA cross-strength rekey support."""
        host = host or self.master
        ver = tasks.get_package_version_and_release(host, 'certmonger')
        if ver.startswith(self._certmonger_cross_rekey_broken_prefix):
            pytest.skip(
                'certmonger %s lacks ML-DSA cross-strength rekey '
                '(needs build with PR #315)' % ver
            )

    def _import_short_lived_service_profile(self, profile_id, days=1):
        """Import a caIPAserviceCert-based profile with short validity.

        Short validity keeps the clock jump for
        :meth:`_simulate_cert_expiry` to ~1 day so IPA subsystem certs
        (multi-year) stay outside certmonger's enroll_ttls window.
        """
        profile_file = '/tmp/%s.cfg' % profile_id
        tasks.kinit_admin(self.master)
        try:
            self.master.run_command([
                'ipa', 'certprofile-show', 'caIPAserviceCert',
                '--out', profile_file,
            ])
            self.master.run_command([
                'sed', '-i',
                '-e', 's/^profileId=.*/profileId=%s/' % profile_id,
                '-e',
                's/^policyset\\.serverCertSet\\.2\\.constraint\\.params'
                '\\.range=.*/policyset.serverCertSet.2.constraint.params'
                '.range=%d/' % (days + 1),
                '-e',
                's/^policyset\\.serverCertSet\\.2\\.default\\.params'
                '\\.range=.*/policyset.serverCertSet.2.default.params'
                '.range=%d/' % days,
                profile_file,
            ])
            result = self.master.run_command([
                'ipa', 'certprofile-import', profile_id,
                '--file', profile_file,
                '--desc', 'Short-lived profile for PQC renewal tests',
                '--store', 'true',
            ], raiseonerr=False)
            if (result.returncode != 0
                    and 'already exists' not in result.stderr_text):
                raise AssertionError(
                    'certprofile-import failed: %s' % result.stderr_text
                )
            self.master.run_command([
                'ipa', 'caacl-add-profile',
                'hosts_services_caIPAserviceCert',
                '--certprofiles', profile_id,
            ], raiseonerr=False)
        finally:
            self.master.run_command(
                ['rm', '-f', profile_file], raiseonerr=False)
            tasks.kdestroy_all(self.master)

    def _cleanup_short_lived_service_profile(self, profile_id):
        tasks.kinit_admin(self.master)
        try:
            self.master.run_command([
                'ipa', 'caacl-del-profile',
                'hosts_services_caIPAserviceCert',
                '--certprofiles', profile_id,
            ], raiseonerr=False)
            self.master.run_command(
                ['ipa', 'certprofile-del', profile_id], raiseonerr=False)
        finally:
            tasks.kdestroy_all(self.master)

    def _simulate_cert_expiry(self, host, certfile, hours_before_expiry=2):
        """Move host clock into certmonger's renew window for *certfile*.

        Stops chronyd so the clock stays put. Caller must restore with
        :meth:`_restore_system_date` in a ``finally`` block.

        :returns: opaque token (UTC timestamp string) for restore
        """
        original = host.run_command(
            ['date', '-u', '+%Y-%m-%d %H:%M:%S']
        ).stdout_text.strip()
        host.run_command(['systemctl', 'stop', 'chronyd'], raiseonerr=False)
        # GNU date: set clock to notAfter - N hours (UTC).
        host.run_command(
            'end=$(openssl x509 -in %s -noout -enddate | cut -d= -f2); '
            'date -u -s "$(date -u -d "$end %d hours ago" '
            '"+%%Y-%%m-%%d %%H:%%M:%%S")"'
            % (certfile, hours_before_expiry)
        )
        host.run_command(['systemctl', 'restart', 'certmonger'])
        return original

    def _restore_system_date(self, host, original_date):
        """Restore clock after :meth:`_simulate_cert_expiry`."""
        host.run_command(['date', '-u', '-s', original_date], raiseonerr=False)
        host.run_command(['systemctl', 'start', 'chronyd'], raiseonerr=False)

    def _enable_certmonger_autorenew(self, host, req_id):
        host.run_command(
            ['getcert', 'start-tracking', '-i', req_id, '-r']
        )

    def _wait_for_cert_serial_change(self, host, req_id, certfile,
                                     old_serial, timeout=300):
        """Wait until certmonger replaces *certfile* with a new serial."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = tasks.wait_for_request(host, req_id, 30)
            if status == 'MONITORING':
                new_serial = self._cert_serial(host, certfile)
                if new_serial != old_serial:
                    return new_serial
            time.sleep(5)
        raise AssertionError(
            'certmonger request %s did not renew within %ss '
            '(serial still %s)' % (req_id, timeout, old_serial)
        )


class TestInstallMasterClientMLDSA(PQCInstallBase,
                                   PQCCertHelpers,
                                   TestInstallMasterClient):
    """ML-DSA IPA service keys with default RSA CA.

    Inherits 20+ existing cert tests from TestInstallMasterClient and adds
    ML-DSA-specific scenarios: user cert requests with ML-DSA CSRs, and
    certmonger host cert enrollment with ML-DSA keys.

    Configuration:
        ipa_key_type: 'mldsa' (default ML-DSA-65)
        ca_key_type: None (RSA CA)
    """

    ipa_key_type = 'mldsa'
    ca_key_type = None
    mldsa_cert_keygen = 'ML-DSA-65'

    # Skip inherited tests that aren't ML-DSA specific or are covered in
    # TestInstallMasterClientMLDSACA (which tests largest cert scenario)
    @pytest.mark.skip(reason="Not ML-DSA specific, covered in parent class")
    def test_certmonger_ipa_responder_jsonrpc(self):
        pass

    @pytest.mark.skip(reason="Large cert handling tested in MLDSA CA variant")
    def test_cacert_file_appear_with_option_F(self):
        pass

    @pytest.mark.skip(reason="SAN with ML-DSA tested in MLDSA CA variant")
    def test_ipa_getcert_san_aci(self):
        pass

    def test_user_cert_mldsa_csr_signed_by_rsa_ca(self):
        """ML-DSA user CSRs are signed by the default RSA CA."""
        self._issue_user_certs(
            'user-mldsa-rsa-ca',
            [('mldsa0', 'mldsa'), ('mldsa1', 'mldsa')],
        )

    def test_getcert_mldsa_key_signed_by_rsa_ca(self):
        """Host cert via ``caIPAserviceCert`` with ML-DSA key, RSA CA."""
        req_id = 'mldsa-host-rsa-ca'
        try:
            certfile = self._getcert_request_host_cert(
                self.master, req_id, keygen=self.mldsa_cert_keygen,
            )
            pk_out = self.master.run_command(
                'openssl x509 -in %s -noout -text | grep Public-Key' % certfile
            ).stdout_text
            assert self.mldsa_cert_keygen in pk_out
        finally:
            self._cleanup_getcert_request(self.master, req_id)


class TestInstallMasterClientMLDSACA(PQCInstallBase,
                                     PQCCertHelpers,
                                     TestInstallMasterClient):
    """ML-DSA IPA service keys and ML-DSA CA signing keys.

    Tests full ML-DSA deployment: both IPA service certs and CA signing
    keys use ML-DSA. Validates mixed cert issuance (RSA CSR → ML-DSA CA,
    ML-DSA CSR → ML-DSA CA), certmonger enrollment from both master and
    client, and explicit ML-DSA rekey/resubmit/expiry autorenew (not
    inherited from RSA TestCertmongerRekey).

    Configuration:
        ipa_key_type: 'mldsa' (default ML-DSA-65)
        ca_key_type: 'mldsa' (default ML-DSA-65)
    """

    ipa_key_type = 'mldsa'
    ca_key_type = 'mldsa'
    mldsa_cert_keygen = 'ML-DSA-65'

    # Skip protocol test - not ML-DSA specific
    @pytest.mark.skip(reason="Not ML-DSA specific, covered in parent class")
    def test_certmonger_ipa_responder_jsonrpc(self):
        pass

    def test_cacert_file_appear_with_option_F(self):
        """Test -F option with ML-DSA CA cert (validates large cert handling).

        ML-DSA CA certificates are significantly larger (~6KB vs ~2KB for RSA).
        This test validates that getcert -F option correctly handles the larger
        CA cert file creation timing.

        Related: https://codeberg.org/freeipa/freeipa/issues/8105
        """
        certfile = os.path.join(paths.OPENSSL_CERTS_DIR, "test.pem")
        keyfile = os.path.join(paths.OPENSSL_PRIVATE_DIR, "test.key")
        cafile = os.path.join(paths.OPENSSL_DIR, "test.CA")

        try:
            # Run parent test (validates -F option behavior)
            super().test_cacert_file_appear_with_option_F()
        finally:
            # Cleanup to avoid conflicts with subsequent test runs
            self.clients[0].run_command(
                ['ipa-getcert', 'stop-tracking', '-f', certfile],
                raiseonerr=False
            )
            self.clients[0].run_command(
                ['rm', '-f', certfile, keyfile, cafile],
                raiseonerr=False
            )

    def test_ipa_getcert_san_aci(self):
        """Test DNS and IP SAN extensions with ML-DSA certs.

        ML-DSA certificates with SAN extensions are larger and encode
        differently than RSA. This validates that large ML-DSA certs with
        SANs are correctly issued and ACIs properly enforced.
        """
        certfile = os.path.join(paths.OPENSSL_CERTS_DIR, "test2.pem")
        keyfile = os.path.join(paths.OPENSSL_PRIVATE_DIR, "test2.key")

        try:
            # Run parent test (validates SAN + ACI behavior)
            super().test_ipa_getcert_san_aci()
        finally:
            # Cleanup certmonger tracking
            self.clients[0].run_command(
                ['ipa-getcert', 'stop-tracking', '-f', certfile],
                raiseonerr=False
            )
            self.clients[0].run_command(
                ['rm', '-f', certfile, keyfile],
                raiseonerr=False
            )

            # Cleanup DNS records
            hostname = self.clients[0].hostname
            tasks.kinit_admin(self.master)
            try:
                zone = tasks.prepare_reverse_zone(
                    self.master, self.clients[0].ip)[0]
                rec = str(self.clients[0].ip).split('.')[3]
                self.master.run_command(
                    ['ipa', 'dnsrecord-del', zone, rec, '--ptr-rec', hostname],
                    raiseonerr=False
                )
            except Exception:
                # DNS cleanup is best-effort
                pass
            finally:
                tasks.kdestroy_all(self.master)

    def test_ca_signing_keys_are_mldsa(self):
        """Dogtag CA signing keys use ML-DSA."""
        result = self.master.run_command(
            'certutil -K -d %s -f %s | grep -c mldsa' % (
                paths.PKI_TOMCAT_ALIAS_DIR,
                paths.PKI_TOMCAT_ALIAS_PWDFILE_TXT,
            )
        )
        key_count = int(result.stdout_text.strip())
        assert key_count == 5, f"Expected 5 ML-DSA CA keys, found {key_count}"

    def test_user_cert_rsa_csr_signed_by_mldsa_ca(self):
        """RSA user CSRs are signed by an ML-DSA CA."""
        self._issue_user_certs(
            'user-rsa-mldsa-ca',
            [('rsa0', 'rsa'), ('rsa1', 'rsa')],
        )

    def test_user_cert_mldsa_csr_signed_by_mldsa_ca(self):
        """ML-DSA user CSRs are signed by an ML-DSA CA."""
        self._issue_user_certs(
            'user-mldsa-mldsa-ca',
            [('mldsa0', 'mldsa'), ('mldsa1', 'mldsa')],
        )

    def test_getcert_mldsa_key_signed_by_mldsa_ca(self):
        """Host cert with ML-DSA key via ML-DSA CA."""
        req_id = 'mldsa-host-mldsa-ca'
        try:
            certfile = self._getcert_request_host_cert(
                self.master, req_id, keygen=self.mldsa_cert_keygen,
            )
            pk_out = self.master.run_command(
                'openssl x509 -in %s -noout -text | grep Public-Key' % certfile
            ).stdout_text
            assert self.mldsa_cert_keygen in pk_out
        finally:
            self._cleanup_getcert_request(self.master, req_id)

    def test_getcert_rsa_key_signed_by_mldsa_ca(self):
        """Default RSA host key enrollment against an ML-DSA CA."""
        req_id = 'rsa-host-mldsa-ca'
        try:
            certfile = self._getcert_request_host_cert(self.master, req_id)
            pk_out = self.master.run_command(
                'openssl x509 -in %s -noout -text | grep Public-Key' % certfile
            ).stdout_text
            assert 'RSA Public-Key' in pk_out or '2048 bit' in pk_out, pk_out
        finally:
            self._cleanup_getcert_request(self.master, req_id)

    def test_getcert_mldsa_key_client_signed_by_mldsa_ca(self):
        """Client host cert with ML-DSA key via ML-DSA CA."""
        self._require_openssl_mldsa(self.clients[0])
        req_id = 'mldsa-client-mldsa-ca'
        host = self.clients[0]
        try:
            certfile = self._getcert_request_host_cert(
                host, req_id, keygen=self.mldsa_cert_keygen,
            )
            pk_out = host.run_command(
                'openssl x509 -in %s -noout -text | grep Public-Key' % certfile
            ).stdout_text
            assert self.mldsa_cert_keygen in pk_out
        finally:
            self._cleanup_getcert_request(host, req_id)

    def test_getcert_rekey_mldsa_same_size(self):
        """Rekey ML-DSA-65 → ML-DSA-65: new key, same strength.

        Explicit PQC coverage parallel to TestCertmongerRekey's RSA
        ``-g`` size change; ML-DSA rekey uses ``-G ML-DSA-65 -g 65``.
        """
        req_id = 'mldsa-rekey-same'
        try:
            certfile = self._getcert_request_host_cert(
                self.master, req_id, keygen=self.mldsa_cert_keygen,
            )
            before_fp = self._cert_pubkey_sha256(self.master, certfile)
            assert self.mldsa_cert_keygen in self._cert_public_key_label(
                self.master, certfile
            )

            self._getcert_rekey(
                self.master, req_id, keygen=self.mldsa_cert_keygen,
            )

            after_fp = self._cert_pubkey_sha256(self.master, certfile)
            pk_out = self._cert_public_key_label(self.master, certfile)
            assert self.mldsa_cert_keygen in pk_out, pk_out
            assert before_fp != after_fp, (
                'rekey should generate a new key pair (pubkey unchanged)'
            )
        finally:
            self._cleanup_getcert_request(self.master, req_id)

    def test_getcert_rekey_mldsa_cross_size(self):
        """Rekey ML-DSA-65 → ML-DSA-87 (needs certmonger with PR #315)."""
        self._require_certmonger_mldsa_cross_rekey()
        target = 'ML-DSA-87'
        self._require_openssl_mldsa(self.master, algorithm=target)
        req_id = 'mldsa-rekey-cross'
        try:
            certfile = self._getcert_request_host_cert(
                self.master, req_id, keygen=self.mldsa_cert_keygen,
            )
            before_fp = self._cert_pubkey_sha256(self.master, certfile)
            assert self.mldsa_cert_keygen in self._cert_public_key_label(
                self.master, certfile
            )

            self._getcert_rekey(self.master, req_id, keygen=target)

            after_fp = self._cert_pubkey_sha256(self.master, certfile)
            pk_out = self._cert_public_key_label(self.master, certfile)
            assert target in pk_out, pk_out
            assert before_fp != after_fp
        finally:
            self._cleanup_getcert_request(self.master, req_id)

    def test_getcert_resubmit_mldsa_preserves_key(self):
        """Resubmit renews the cert but preserves the ML-DSA key pair."""
        req_id = 'mldsa-resubmit-key'
        try:
            certfile = self._getcert_request_host_cert(
                self.master, req_id, keygen=self.mldsa_cert_keygen,
            )
            before_fp = self._cert_pubkey_sha256(self.master, certfile)
            before_key = self._keyfile_sha256(self.master, req_id)
            before_serial = self._cert_serial(self.master, certfile)

            self._getcert_resubmit(self.master, req_id)

            after_fp = self._cert_pubkey_sha256(self.master, certfile)
            after_key = self._keyfile_sha256(self.master, req_id)
            after_serial = self._cert_serial(self.master, certfile)
            pk_out = self._cert_public_key_label(self.master, certfile)
            assert self.mldsa_cert_keygen in pk_out, pk_out
            assert before_fp == after_fp, (
                'resubmit must preserve the existing ML-DSA public key'
            )
            assert before_key == after_key, (
                'resubmit must preserve the existing ML-DSA private key file'
            )
            assert before_serial != after_serial, (
                'resubmit should issue a new certificate (serial unchanged)'
            )
        finally:
            self._cleanup_getcert_request(self.master, req_id)

    def test_getcert_rekey_mldsa_request_id(self):
        """Rename request id via rekey ``-I`` for an ML-DSA tracked cert."""
        req_id = 'mldsa-rekey-id'
        new_req_id = 'mldsa-rekey-id-new'
        try:
            self._getcert_request_host_cert(
                self.master, req_id, keygen=self.mldsa_cert_keygen,
            )
            result = self._getcert_rekey(
                self.master, req_id, new_id=new_req_id,
            )
            assert new_req_id in result.stdout_text

            # Rename back so cleanup uses the original id paths/tracking.
            result = self._getcert_rekey(
                self.master, new_req_id, new_id=req_id,
            )
            assert req_id in result.stdout_text
        finally:
            self._cleanup_getcert_request(self.master, req_id)
            self._cleanup_getcert_request(self.master, new_req_id)

    def test_getcert_mldsa_autorenew_on_expiry(self):
        """Simulate near-expiry; certmonger autorenews keeping the ML-DSA key.

        Uses a 1-day profile so the clock jump stays small and IPA
        subsystem certs are not pulled into certmonger's renew window.
        Autorenew is disabled at request time, then enabled after the
        serial/pubkey snapshot so issuance itself cannot race renewal.
        """
        profile = 'caIPAserviceCertPQCShort'
        req_id = 'mldsa-expiry-renew'
        restore = None
        try:
            self._import_short_lived_service_profile(profile, days=1)
            certfile = self._getcert_request_host_cert(
                self.master, req_id,
                keygen=self.mldsa_cert_keygen,
                profile=profile,
                auto_renew=False,
            )
            before_fp = self._cert_pubkey_sha256(self.master, certfile)
            before_serial = self._cert_serial(self.master, certfile)
            assert self.mldsa_cert_keygen in self._cert_public_key_label(
                self.master, certfile
            )

            self._enable_certmonger_autorenew(self.master, req_id)
            restore = self._simulate_cert_expiry(
                self.master, certfile, hours_before_expiry=2,
            )
            self._wait_for_cert_serial_change(
                self.master, req_id, certfile, before_serial,
            )

            after_fp = self._cert_pubkey_sha256(self.master, certfile)
            after_serial = self._cert_serial(self.master, certfile)
            pk_out = self._cert_public_key_label(self.master, certfile)
            assert self.mldsa_cert_keygen in pk_out, pk_out
            assert before_fp == after_fp, (
                'autorenew on expiry must preserve the ML-DSA public key'
            )
            assert before_serial != after_serial
        finally:
            if restore is not None:
                self._restore_system_date(self.master, restore)
            self._cleanup_getcert_request(self.master, req_id)
            self._cleanup_short_lived_service_profile(profile)


class PQCEnrollmentInstall:
    """PQC install for master + replica (enrollment-focused topology).

    Inherited from IntegrationTest (via multiple inheritance):
        master, replicas: Test host instances
    """

    num_replicas = 1
    master_with_dns = True
    ipa_key_type = None
    ca_key_type = None
    mldsa_cert_keygen = 'ML-DSA-65'

    @classmethod
    def install(cls, mh):
        extra_args = []
        if cls.ipa_key_type:
            extra_args.extend(['--key-type-size', cls.ipa_key_type])
        if cls.ca_key_type:
            extra_args.extend(['--ca-key-type', cls.ca_key_type])
        tasks.install_master(
            cls.master, setup_dns=True,
            extra_args=extra_args
        )
        tasks.install_replica(
            cls.master, cls.replicas[0],
            setup_ca=True,
        )

    def _cleanup_mldsa_enroll_files(self, host, req_id):
        certfile = os.path.join(paths.OPENSSL_CERTS_DIR, '%s.pem' % req_id)
        keyfile = os.path.join(paths.OPENSSL_PRIVATE_DIR, '%s.key' % req_id)
        host.run_command(
            ['getcert', 'stop-tracking', '-i', req_id], raiseonerr=False)
        host.run_command(['rm', '-f', certfile, keyfile], raiseonerr=False)

    def _request_mldsa_caIPAservice_cert(self, host, req_id):
        """Issue a host cert via Dogtag profile ``caIPAserviceCert``."""
        certfile = os.path.join(paths.OPENSSL_CERTS_DIR, '%s.pem' % req_id)
        keyfile = os.path.join(paths.OPENSSL_PRIVATE_DIR, '%s.key' % req_id)
        hostname = host.hostname
        host.run_command(['rm', '-f', certfile, keyfile], raiseonerr=False)
        cmd_arg = [
            'getcert', 'request', '-c', 'ipa', '-I', req_id,
            '-k', keyfile, '-f', certfile,
            '-D', hostname, '-K', 'host/%s' % hostname,
            '-N', 'CN={}'.format(hostname),
            '-U', 'id-kp-clientAuth', '-T', 'caIPAserviceCert',
            '-G', self.mldsa_cert_keygen,
        ]
        result = host.run_command(cmd_arg)
        assert (
            'New signing request "%s" added.\n' % req_id in result.stdout_text
        )
        status = tasks.wait_for_request(host, req_id, 300)
        assert status == 'MONITORING', (
            'certmonger request %s is in state %s' % (req_id, status)
        )
        list_out = host.run_command(
            ['getcert', 'list', '-i', req_id]
        ).stdout_text
        assert 'profile: caIPAserviceCert' in list_out
        pk_out = host.run_command(
            'openssl x509 -in %s -noout -text | grep Public-Key' % certfile
        ).stdout_text
        assert self.mldsa_cert_keygen in pk_out

    def _request_mldsa_via_ipa_cert_request(self, host, stem, service_suffix):
        """Issue a cert with ``ipa cert-request`` (PKCS#10 CSR)."""
        csr = os.path.join(paths.OPENSSL_CERTS_DIR, '%s.csr' % stem)
        key = os.path.join(paths.OPENSSL_PRIVATE_DIR, '%s.key' % stem)
        cert = os.path.join(paths.OPENSSL_CERTS_DIR, '%s.crt' % stem)
        algo = self.mldsa_cert_keygen
        principal = 'mldsaenr{}/{}'.format(service_suffix, host.hostname)

        host.run_command(['rm', '-f', csr, key, cert], raiseonerr=False)
        gen = host.run_command(
            ['openssl', 'genpkey', '-algorithm', algo, '-out', key],
            raiseonerr=False,
        )
        if gen.returncode != 0:
            host.run_command(['rm', '-f', key], raiseonerr=False)
            pytest.skip(
                'OpenSSL on %s cannot generate %s keys (need OpenSSL with '
                'ML-DSA support).' % (host.hostname, algo)
            )
        cn = host.hostname
        try:
            host.run_command([
                'openssl', 'req', '-new', '-key', key, '-out', csr,
                '-subj', '/CN={}'.format(cn),
            ])
            tasks.kinit_admin(host)
            try:
                host.run_command([
                    'ipa', 'cert-request', '--add', '--principal', principal,
                    '--certificate-out', cert, '--profile-id',
                    'caIPAserviceCert', csr,
                ])
            finally:
                tasks.kdestroy_all(host)

            pk_out = host.run_command(
                'openssl x509 -in %s -noout -text | grep Public-Key' % cert
            ).stdout_text
            assert algo in pk_out
        finally:
            host.run_command(
                ['ipa', 'service-del', principal], raiseonerr=False)
            host.run_command(['rm', '-f', csr, key, cert], raiseonerr=False)


class TestPQCCertEnrollmentIPACerts(PQCEnrollmentInstall,
                                    IntegrationTest):
    """Certmonger enrollment with ML-DSA service keys (RSA CA)."""

    ipa_key_type = 'mldsa'
    ca_key_type = None
    mldsa_cert_keygen = 'ML-DSA-65'

    def test_getcert_enroll_caIPAserviceCert_mldsa_key_master(self):
        req_id = 'pqc-mldsa-enroll-master'
        try:
            self._request_mldsa_caIPAservice_cert(self.master, req_id)
        finally:
            self._cleanup_mldsa_enroll_files(self.master, req_id)

    def test_getcert_enroll_caIPAserviceCert_mldsa_key_replica(self):
        req_id = 'pqc-mldsa-enroll-replica'
        try:
            self._request_mldsa_caIPAservice_cert(self.replicas[0], req_id)
        finally:
            self._cleanup_mldsa_enroll_files(self.replicas[0], req_id)

    def test_ipa_cert_request_mldsa_caIPAservice_profile_master(self):
        suffix = ''.join(
            random.choice(string.ascii_lowercase) for _ in range(8)
        )
        stem = 'pqc-ipa-cr-%s' % suffix
        self._request_mldsa_via_ipa_cert_request(self.master, stem, suffix)


class TestPQCCertEnrollmentCACerts(PQCEnrollmentInstall,
                                   IntegrationTest):
    """Enrollment when both IPA service keys and the CA use ML-DSA."""

    ipa_key_type = 'mldsa:44'
    ca_key_type = 'mldsa'
    mldsa_cert_keygen = 'ML-DSA-44'

    def test_getcert_enroll_caIPAserviceCert_mldsa_key_master(self):
        req_id = 'pqc-mldsa44-enroll-master'
        try:
            self._request_mldsa_caIPAservice_cert(self.master, req_id)
        finally:
            self._cleanup_mldsa_enroll_files(self.master, req_id)

    def test_getcert_enroll_caIPAserviceCert_mldsa_key_replica(self):
        req_id = 'pqc-mldsa44-enroll-replica'
        try:
            self._request_mldsa_caIPAservice_cert(self.replicas[0], req_id)
        finally:
            self._cleanup_mldsa_enroll_files(self.replicas[0], req_id)

    def test_ipa_cert_request_mldsa_caIPAservice_profile_master(self):
        suffix = ''.join(
            random.choice(string.ascii_lowercase) for _ in range(8)
        )
        stem = 'pqc-ipa-cr44-%s' % suffix
        self._request_mldsa_via_ipa_cert_request(self.master, stem, suffix)

import filecmp
import os
import shutil
import subprocess
import unittest

# Named import, so that Pytest does not pick up TC_00_FileSystem as belonging to this module.
import test_fs

from graminelibos import _CONFIG_SGX_ENABLED

# TODO: While encrypted files are no longer SGX-only, the related tools (gramine-sgx-pf-crypt,
# gramine-sgx-pf-tamper) are still part of Linux-SGX PAL. As a result, we are able to run the tests
# with other PALs, but only if Gramine was built with SGX enabled.

@unittest.skipUnless(_CONFIG_SGX_ENABLED, 'Encrypted files tests require SGX to be enabled')
class TC_50_EncryptedFiles(test_fs.TC_00_FileSystem):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.PF_CRYPT = 'gramine-sgx-pf-crypt'
        cls.PF_TAMPER = 'gramine-sgx-pf-tamper'
        cls.WRAP_KEY = os.path.join(cls.TEST_DIR, 'wrap-key')
        # CONST_WRAP_KEY must match the one in manifest
        cls.CONST_WRAP_KEY = [0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
                              0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00]
        cls.ENCRYPTED_DIR = os.path.join(cls.TEST_DIR, 'enc_input')
        cls.ENCRYPTED_FILES = [os.path.join(cls.ENCRYPTED_DIR, str(v)) for v in cls.FILE_SIZES]
        cls.LIB_PATH = os.path.join(os.getcwd(), 'lib')

        if not os.path.exists(cls.ENCRYPTED_DIR):
            os.mkdir(cls.ENCRYPTED_DIR)
        cls.OUTPUT_DIR = os.path.join(cls.TEST_DIR, 'enc_output')
        cls.OUTPUT_FILES = [os.path.join(cls.OUTPUT_DIR, str(x)) for x in cls.FILE_SIZES]
        # create encrypted files
        cls.__set_default_key(cls)
        for i in cls.INDEXES:
            cmd = [cls.PF_CRYPT, 'encrypt', '-w', cls.WRAP_KEY, '-i', cls.INPUT_FILES[i], '-o',
                   cls.ENCRYPTED_FILES[i]]

            cls.run_native_binary(cmd)

    def __pf_crypt(self, args):
        args.insert(0, self.PF_CRYPT)
        return self.run_native_binary(args)

    def __set_default_key(self):
        with open(self.WRAP_KEY, 'wb') as file:
            file.write(bytes(self.CONST_WRAP_KEY))

    # overrides TC_00_FileSystem to encrypt the file instead of just copying
    def copy_input(self, input_path, output_path):
        self.__encrypt_file(input_path, output_path)

    def __encrypt_file(self, input_path, output_path):
        args = ['encrypt', '-w', self.WRAP_KEY, '-i', input_path, '-o', output_path]
        stdout, stderr = self.__pf_crypt(args)
        return (stdout, stderr)

    def __decrypt_file(self, input_path, output_path):
        args = ['decrypt', '-w', self.WRAP_KEY, '-i', input_path, '-o', output_path]
        stdout, stderr = self.__pf_crypt(args)
        return (stdout, stderr)

    def test_000_gen_key(self):
        # test random key generation
        key_path = os.path.join(self.TEST_DIR, 'tmpkey')
        args = ['gen-key', '-w', key_path]
        stdout, _ = self.__pf_crypt(args)
        self.assertIn('Wrap key saved to: ' + key_path, stdout)
        self.assertEqual(os.path.getsize(key_path), 16)
        os.remove(key_path)

    def test_010_encrypt_decrypt(self):
        for i in self.INDEXES:
            self.__encrypt_file(self.INPUT_FILES[i], self.OUTPUT_FILES[i])
            self.assertFalse(filecmp.cmp(self.INPUT_FILES[i], self.OUTPUT_FILES[i], shallow=False))
            dec_path = os.path.join(self.OUTPUT_DIR,
                                    os.path.basename(self.OUTPUT_FILES[i]) + '.dec')
            self.__decrypt_file(self.OUTPUT_FILES[i], dec_path)
            self.assertTrue(filecmp.cmp(self.INPUT_FILES[i], dec_path, shallow=False))

    # overrides TC_00_FileSystem to change input dir (from plaintext to encrypted)
    def test_100_open_close(self):
        input_path = self.ENCRYPTED_FILES[-1] # existing file
        output_path = os.path.join(self.OUTPUT_DIR, 'test_100') # new file
        stdout, stderr = self.run_binary(['open_close', 'R', input_path])
        self.verify_open_close(stdout, stderr, input_path, 'input')
        stdout, stderr = self.run_binary(['open_close', 'W', output_path])
        self.verify_open_close(stdout, stderr, output_path, 'output')
        self.assertTrue(os.path.isfile(output_path))

    # overrides TC_00_FileSystem to change input dir (from plaintext to encrypted)
    # doesn't work because encrypted files do not support truncation (and the test opens an
    # existing, non-empty file with O_TRUNC)
    @unittest.expectedFailure
    def test_101_open_flags(self):
        # the test binary expects a path to file that will get created
        file_path = os.path.join(self.OUTPUT_DIR, 'test_101') # new file
        stdout, stderr = self.run_binary(['open_flags', file_path])
        self.verify_open_flags(stdout, stderr)

    # overrides TC_00_FileSystem to change input dir (from plaintext to encrypted)
    def test_115_seek_tell(self):
        # the test binary expects a path to read-only (existing) file and two paths to files that
        # will get created
        plaintext_path = self.INPUT_FILES[-1]
        input_path = self.ENCRYPTED_FILES[-1] # existing file
        output_path_1 = os.path.join(self.OUTPUT_DIR, 'test_115a') # writable files
        output_path_2 = os.path.join(self.OUTPUT_DIR, 'test_115b')
        self.copy_input(plaintext_path, output_path_1) # encrypt
        self.copy_input(plaintext_path, output_path_2)
        stdout, stderr = self.run_binary(['seek_tell', input_path, output_path_1, output_path_2])
        self.verify_seek_tell(stdout, stderr, input_path, output_path_1, output_path_2,
                              self.FILE_SIZES[-1])

    # overrides TC_00_FileSystem to change input dir (from plaintext to encrypted)
    def test_130_file_stat(self):
        # the test binary expects a path to read-only (existing) file and a path to file that
        # will get created
        for i in self.INDEXES:
            input_path = self.ENCRYPTED_FILES[i]
            output_path = self.OUTPUT_FILES[i]
            size = str(self.FILE_SIZES[i])
            self.copy_input(self.INPUT_FILES[i], output_path)
            stdout, stderr = self.run_binary(['stat', input_path, output_path])
            self.verify_stat(stdout, stderr, input_path, output_path, size)

    # overrides TC_00_FileSystem to decrypt output
    def verify_size(self, file, size):
        dec_path = os.path.join(self.OUTPUT_DIR, os.path.basename(file) + '.dec')
        self.__decrypt_file(file, dec_path)
        self.assertEqual(os.stat(dec_path).st_size, size)

    @unittest.expectedFailure
    # pylint: disable=fixme
    def test_140_file_truncate(self):
        self.fail() # TODO: port these to the new file format

    def test_150_file_rename(self):
        path1 = os.path.join(self.OUTPUT_DIR, 'test_150a')
        path2 = os.path.join(self.OUTPUT_DIR, 'test_150b')
        self.copy_input(self.ENCRYPTED_FILES[-1], path1)
        shutil.copy(path1, path2)
        # accessing renamed file should fail
        args = ['decrypt', '-V', '-w', self.WRAP_KEY, '-i', path2, '-o', path1]
        try:
            self.__pf_crypt(args)
        except subprocess.CalledProcessError as exc:
            self.assertNotEqual(exc.returncode, 0)
        else:
            print('[!] Fail: successfully decrypted renamed file: ' + path2)
            self.fail()

    # overrides TC_00_FileSystem to decrypt output
    def verify_copy_content(self, input_path, output_path):
        dec_path = os.path.join(self.OUTPUT_DIR, os.path.basename(output_path) + '.dec')
        self.__decrypt_file(output_path, dec_path)
        self.assertTrue(filecmp.cmp(input_path, dec_path, shallow=False))

    # overrides TC_00_FileSystem to change input dir (from plaintext to encrypted)
    def do_copy_test(self, executable, timeout):
        stdout, stderr = self.run_binary([executable, self.ENCRYPTED_DIR, self.OUTPUT_DIR],
                                         timeout=timeout)
        self.verify_copy(stdout, stderr, self.ENCRYPTED_DIR, executable)

    # overrides TC_00_FileSystem to not skip this on SGX
    def test_204_copy_dir_mmap_whole(self):
        self.do_copy_test('copy_mmap_whole', 30)

    # overrides TC_00_FileSystem to not skip this on SGX
    def test_205_copy_dir_mmap_seq(self):
        self.do_copy_test('copy_mmap_seq', 60)

    # overrides TC_00_FileSystem to not skip this on SGX
    def test_206_copy_dir_mmap_rev(self):
        self.do_copy_test('copy_mmap_rev', 60)

    # overrides TC_00_FileSystem to change dirs (from plaintext to encrypted)
    def test_210_copy_dir_mounted(self):
        executable = 'copy_whole'
        stdout, stderr = self.run_binary([executable, '/mounted/enc_input', '/mounted/enc_output'],
                                         timeout=30)
        self.verify_copy(stdout, stderr, '/mounted/enc_input', executable)

    def __corrupt_file(self, input_path, output_path):
        cmd = [self.PF_TAMPER, '-w', self.WRAP_KEY, '-i', input_path, '-o', output_path]
        return self.run_native_binary(cmd)

    # invalid/corrupted files
    def test_500_invalid(self):
        invalid_dir = os.path.join(self.TEST_DIR, 'enc_invalid')
        if not os.path.exists(invalid_dir):
            os.mkdir(invalid_dir)

        # prepare valid encrypted file (largest one for maximum possible corruptions)
        original_input = self.OUTPUT_FILES[-1]
        self.__encrypt_file(self.INPUT_FILES[-1], original_input)
        # generate invalid files based on the above
        self.__corrupt_file(original_input, invalid_dir)

        # try to decrypt invalid files
        for name in os.listdir(invalid_dir):
            invalid = os.path.join(invalid_dir, name)
            output_path = os.path.join(self.OUTPUT_DIR, name)
            input_path = os.path.join(invalid_dir, os.path.basename(original_input))
            # copy the file so it has the original file name (for allowed path check)
            shutil.copy(invalid, input_path)

            try:
                args = ['decrypt', '-V', '-w', self.WRAP_KEY, '-i', input_path, '-o', output_path]
                self.__pf_crypt(args)
            except subprocess.CalledProcessError as exc:
                # decryption of invalid file must fail with -1 (wrapped to 255)
                self.assertEqual(exc.returncode, 255)
            else:
                print('[!] Fail: successfully decrypted file: ' + name)
                self.fail()

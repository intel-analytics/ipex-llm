Test purpose
------------

These tests perform common FS operations in various ways to exercise the Gramine
FS subsystem:

- open/close
- read/write
- create/delete
- read/change size
- seek/tell
- memory-mapped read/write
- sendfile
- copy directory in different ways

How to execute
--------------

- `gramine-test pytest -v`

Encrypted file tests assume that Gramine was built with SGX enabled (see comment
in `test_enc.py`).

```
cd $gramine/Pal/src/host/Linux-SGX/tools
make install PREFIX=$gramine/LibOS/shim/test/fs
```

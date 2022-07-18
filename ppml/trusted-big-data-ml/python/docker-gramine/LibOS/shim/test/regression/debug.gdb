set breakpoint pending on
set pagination off

# Check if debug sources are loaded in our program, and we can break inside.

tbreak func
commands
  echo \n<backtrace 1 start>\n
  backtrace
  echo <backtrace 1 end>\n\n

  # Check if we can break inside PAL and get a full backtrace (across libc).

  tbreak dev_write
  commands
    echo \n<backtrace 2 start>\n
    backtrace
    echo <backtrace 2 end>\n\n
    continue
  end

  # Check if we can break inside OCALL (SGX only).

  tbreak sgx_ocall_write
  commands
    echo \n<backtrace 3 start>\n
    backtrace
    echo <backtrace 3 end>\n\n
    continue
  end

  continue
end

run

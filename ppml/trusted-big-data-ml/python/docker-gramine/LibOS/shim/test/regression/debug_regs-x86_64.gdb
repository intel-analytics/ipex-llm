set breakpoint pending on
set pagination off

# Run until trap #1
run

# Run until trap #2, check RDX register read
set val = 0x1000100010001000
continue
echo \n<RDX start>\n
print/x $rdx
echo <RDX end>\n\n

# Check RDX register write
set $rdx = 0x2000200020002000
continue
echo \n<RDX result start>\n
print/x val
echo <RDX result end>\n\n

# Run until trap #3, check XMM0 register read
set val = 0x3000300030003000
continue
echo \n<XMM0 start>\n
print/x $xmm0.uint128
echo <XMM0 end>\n\n

# Check XMM0 register write
set $xmm0.uint128 = 0x4000400040004000
continue
echo \n<XMM0 result start>\n
print/x val
echo <XMM0 result end>\n\n

# Run until end
continue

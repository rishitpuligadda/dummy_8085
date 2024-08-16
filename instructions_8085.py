import pdb
# Define global CPU state
memory = [0] * 65536  # Memory space (64KB)
stack_pointer = 0xFFFF  # Stack pointer initialized to the top of memory
program_counter = 0x0000  # Program counter

register_code = {
    'A': 0x07,
    'B': 0x00,
    'C': 0x01,
    'D': 0x02,
    'E': 0x03,
    'H': 0x04,
    'L': 0x05,
    'M': 0x06,  # Memory reference (HL register pair)
}

code_register = {
    0x07 : 'A',
    0x00 : 'B',
    0x01 : 'C',
    0x02 : 'D',
    0x03 : 'E',
    0x04 : 'H',
    0x05 : 'L',
    0x06 : 'M',
}

flags = {
    'Z': 0,
    'S': 0,
    'P': 0,
    'CY': 0,
    'AC': 0
}

registers = {
    'A': 0x00,
    'B': 0x00,
    'C': 0x00,
    'D': 0x00,
    'E': 0x00,
    'H': 0x00,
    'L': 0x00,
}

def reg_pair(rp):
    reg_high = register_code[rp]
    rp = [code_register[reg_high], code_register[reg_high+1]]
    return rp
# Data Transfer Instructions

def mov(dest, src):
    """Move data from source to destination."""
    registers[dest] = registers[src]

def mvi(reg, data):
    """Move immediate data to register."""
    registers[reg] = data

def lxi(rp, highByte, lowByte):
    """Load immediate 16-bit data into a register pair."""
    rp = reg_pair(rp)
    registers[rp[0]] = highByte
    registers[rp[1]] = lowByte

def lda(addr16):
    """Load accumulator with the value at address."""
    registers['A'] = memory[addr16]

def sta(addr16):
    """Store accumulator value at address."""
    memory[addr16] = registers['A']

def lhld(addr16):
    """Load H and L registers with data from address."""
    registers['L'] = memory[addr16]
    registers['H'] = memory[addr16 + 1]

def shld(addr16):
    """Store H and L register values at address."""
    memory[addr16] = registers['L']
    memory[addr16 + 1] = registers['H']

def cpi(data):
    """Compare immediate data with accumulator."""
    result = registers['A'] - data
    flags['Z'] = int(result == 0)
    flags['CY'] = int(result < 0)
    flags['S'] = int(result < 0)
    flags['P'] = int(bin(result & 0xFF).count('1') % 2 == 0)  # Even parity

def rim():
    """Read interrupt mask."""
    # Example implementation, assuming specific interrupt handling
    pass

def sim():
    """Set interrupt mask."""
    # Example implementation, assuming specific interrupt handling
    pass

# Arithmetic Instructions

def add(reg):
    """Add register value to accumulator."""
    result = registers['A'] + registers[reg]
    registers['A'] = result & 0xFF  # 8-bit result
    flags['CY'] = int(result > 0xFF)
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)  # Sign flag

def adi(data):
    """Add immediate data to accumulator."""
    result = registers['A'] + data
    registers['A'] = result & 0xFF
    flags['CY'] = int(result > 0xFF)
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)

def sub(reg):
    """Subtract register value from accumulator."""
    result = registers['A'] - registers[reg]
    registers['A'] = result & 0xFF
    flags['CY'] = int(result < 0)
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)

def sbi(data):
    """Subtract immediate data from accumulator with borrow."""
    result = registers['A'] - data - flags['CY']
    registers['A'] = result & 0xFF
    flags['CY'] = int(result < 0)
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)

def incr(reg):
    """Increment the value of a register."""
    result = registers[reg] + 1
    registers[reg] = result & 0xFF
    flags['Z'] = int(registers[reg] == 0)
    flags['S'] = int(registers[reg] & 0x80 != 0)

def dcr(reg):
    """Decrement the value of a register."""
    result = registers[reg] - 1
    registers[reg] = result & 0xFF
    flags['Z'] = int(registers[reg] == 0)
    flags['S'] = int(registers[reg] & 0x80 != 0)
"""
def rlc():
    Rotate accumulator left through carry.
    result = (registers['A'] << 1) | flags['CY']
    flags['CY'] = (registers['A'] >> 7) & 1
    registers['A'] = result & 0xFF
"""

def rlc():
    """Rotate accumulator left through carry."""
    carry_out = (registers['A'] >> 7) & 1
    result = (registers['A'] << 1) | flags['CY']
    flags['CY'] = carry_out
    registers['A'] = result & 0xFF

def rrc():
    """Rotate accumulator right through carry."""
    result = (registers['A'] >> 1) | (flags['CY'] << 7)
    flags['CY'] = registers['A'] & 1
    registers['A'] = result & 0xFF

def ral():
    """Rotate accumulator left."""
    result = (registers['A'] << 1) | flags['CY']
    flags['CY'] = (registers['A'] >> 7) & 1
    registers['A'] = result & 0xFF

def rar():
    """Rotate accumulator right."""
    result = (registers['A'] >> 1) | (flags['CY'] << 7)
    flags['CY'] = registers['A'] & 1
    registers['A'] = result & 0xFF

def cma():
    """Complement accumulator."""
    registers['A'] = ~registers['A'] & 0xFF

def cmc():
    """Complement carry flag."""
    flags['CY'] = 1 - flags['CY']

def stc():
    """Set carry flag."""
    flags['CY'] = 1

def cfc():
    """Clear carry flag."""
    flags['CY'] = 0

# Logic Instructions

def ana(reg):
    """Logical AND register value with accumulator."""
    registers['A'] &= registers[reg]
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)
    flags['P'] = int(bin(registers['A']).count('1') % 2 == 0)  # Even parity

def ani(data):
    """Logical AND immediate data with accumulator."""
    registers['A'] &= data
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)
    flags['P'] = int(bin(registers['A']).count('1') % 2 == 0)

def xra(reg):
    """Logical XOR register value with accumulator."""
    registers['A'] ^= registers[reg]
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)
    flags['P'] = int(bin(registers['A']).count('1') % 2 == 0)

def xri(data):
    """Logical XOR immediate data with accumulator."""
    registers['A'] ^= data
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)
    flags['P'] = int(bin(registers['A']).count('1') % 2 == 0)

def ora(reg):
    """Logical OR register value with accumulator."""
    registers['A'] |= registers[reg]
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)
    flags['P'] = int(bin(registers['A']).count('1') % 2 == 0)

def ori(data):
    """Logical OR immediate data with accumulator."""
    registers['A'] |= data
    flags['Z'] = int(registers['A'] == 0)
    flags['S'] = int(registers['A'] & 0x80 != 0)
    flags['P'] = int(bin(registers['A']).count('1') % 2 == 0)

def cpi(data):
    """Compare immediate data with accumulator."""
    result = registers['A'] - data
    flags['Z'] = int(result == 0)
    flags['CY'] = int(result < 0)
    flags['S'] = int(result < 0)
    flags['P'] = int(bin(result & 0xFF).count('1') % 2 == 0)  # Even parity

# Control Instructions

def nop():
    """No operation."""
    pass

def halt():
    """Halt the CPU."""
    # Example implementation, assuming specific interrupt handling
    pass

def di():
    """Disable interrupts."""
    # Example implementation, assuming specific interrupt handling
    pass

def ei():
    """Enable interrupts."""
    # Example implementation, assuming specific interrupt handling
    pass

def rlc():
    """Rotate accumulator left through carry."""
    result = (registers['A'] << 1) | flags['CY']
    flags['CY'] = (registers['A'] >> 7) & 1
    registers['A'] = result & 0xFF

def rrc():
    """Rotate accumulator right through carry."""
    result = (registers['A'] >> 1) | (flags['CY'] << 7)
    flags['CY'] = registers['A'] & 1
    registers['A'] = result & 0xFF

def ral():
    """Rotate accumulator left."""
    result = (registers['A'] << 1) | flags['CY']
    flags['CY'] = (registers['A'] >> 7) & 1
    registers['A'] = result & 0xFF

def rar():
    """Rotate accumulator right."""
    result = (registers['A'] >> 1) | (flags['CY'] << 7)
    flags['CY'] = registers['A'] & 1
    registers['A'] = result & 0xFF

def cma():
    """Complement accumulator."""
    registers['A'] = ~registers['A'] & 0xFF

def cmc():
    """Complement carry flag."""
    flags['CY'] = 1 - flags['CY']

def stc():
    """Set carry flag."""
    flags['CY'] = 1

def cfc():
    """Clear carry flag."""
    flags['CY'] = 0

def rlc():
    """Rotate accumulator left through carry."""
    result = (registers['A'] << 1) | flags['CY']
    flags['CY'] = (registers['A'] >> 7) & 1
    registers['A'] = result & 0xFF

def rrc():
    """Rotate accumulator right through carry."""
    result = (registers['A'] >> 1) | (flags['CY'] << 7)
    flags['CY'] = registers['A'] & 1
    registers['A'] = result & 0xFF

# Branch Instructions

def call(addr16):
    """Call subroutine at address."""
    global stack_pointer, program_counter
    memory[stack_pointer] = (program_counter >> 8) & 0xFF
    stack_pointer -= 1
    memory[stack_pointer] = program_counter & 0xFF
    stack_pointer -= 1
    program_counter = addr16

def ret():
    """Return from subroutine."""
    global stack_pointer, program_counter
    program_counter = (memory[stack_pointer + 1] << 8) | memory[stack_pointer]
    stack_pointer += 2

def rz():
    """Return if zero flag is set."""
    if flags['Z']:
        ret()

def rnz():
    """Return if zero flag is not set."""
    if not flags['Z']:
        ret()

def rp():
    """Return if positive flag is set."""
    if not flags['S']:
        ret()

def rm():
    """Return if minus flag is set."""
    if flags['S']:
        ret()

def rnc():
    """Return if carry flag is not set."""
    if not flags['CY']:
        ret()

def rc():
    """Return if carry flag is set."""
    if flags['CY']:
        ret()

def rpo():
    """Return if parity odd flag is set."""
    if not flags['P']:
        ret()

def rpe():
    """Return if parity even flag is set."""
    if flags['P']:
        ret()

# Stack Instructions

def push(rp):
    """Push register pair onto stack."""
    global stack_pointer
    memory[stack_pointer] = registers[rp] >> 8
    stack_pointer -= 1
    memory[stack_pointer] = registers[rp] & 0xFF
    stack_pointer -= 1

def pop(rp):
    """Pop register pair from stack."""
    global stack_pointer
    registers[rp] = (memory[stack_pointer + 1] << 8) | memory[stack_pointer]
    stack_pointer += 2

def xthl():
    """Exchange top of stack with H and L registers."""
    global stack_pointer
    temp = memory[stack_pointer]
    memory[stack_pointer] = registers['L']
    registers['L'] = temp
    temp = memory[stack_pointer + 1]
    memory[stack_pointer + 1] = registers['H']
    registers['H'] = temp

def sp_hl():
    """Load stack pointer into HL register pair."""
    registers['H'] = stack_pointer >> 8
    registers['L'] = stack_pointer & 0xFF

# I/O Instructions

def in_(port):
    """Read input from port."""
    # Example implementation
    pass

def out(port):
    """Output to port."""
    # Example implementation
    pass


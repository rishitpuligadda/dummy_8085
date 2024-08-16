from tabulate import tabulate
import logging
import pdb
from assembler_trace import *
from instructions_8085 import *

# Configure the logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Create a logger object
logger = logging.getLogger('assembler-8085')

"""
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
"""

opcode_instruction_map = {
    # Data Transfer Instructions
    0x40: "MOV",
    0x06: "MVI",
    0x01: "LXI",
    0xfa: "LDA",
    0xea: "STA",
    0x2a: "LHLD",
    0x22: "SHLD",
    0xfe: "CPI",
    0x20: "RIM",
    0x30: "SIM",

    # Arithmetic Instructions
    0x80: "ADD",
    0xc6: "ADI",
    0x90: "SUB",
    0xd6: "SBI",
    0x04: "INR",
    0x05: "DCR",

    # Logic Instructions
    0xa0: "ANA",
    0xe6: "ANI",
    0xa8: "XRA",
    0xee: "XRI",
    0xb0: "ORA",
    0xf6: "ORI",

    # Control Instructions
    0x00: "NOP",
    0x76: "HALT",
    0xf3: "DI",
    0xfb: "EI",
    0x07: "RLC",
    0x0f: "RRC",
    0x17: "RAL",
    0x1f: "RAR",
    0x2f: "CMA",
    0x3f: "CMC",
    0x37: "STC",

    # Branch Instructions
    0xc3: "JMP",
    0xcd: "CALL",
    0xc9: "RET",
    0xc8: "RZ",
    0xc0: "RNZ",
    0xf0: "RP",
    0xf8: "RM",
    0xd0: "RNC",
    0xd8: "RC",
    0xca: "JZ",
    0xc2: "JNZ",
    0xe9: "JP",
    0xea: "JM",
    0xda: "JC",
    0xd2: "JNC",
    0xe3: "JPO",
    0xe2: "JPE",

    # Stack Instructions
    0xc5: "PUSH",
    0xc1: "POP",
    0xe3: "XTHL",
    0xf9: "SPHL",

    # I/O Instructions
    0xdb: "IN",
    0xd3: "OUT",
}

instruction_metadata = {
    # Data Transfer Instructions
    'MOV': {'opcode': 0x40, 'bytes': 2, 'operands': ['reg', 'reg'], 'execute': lambda dest, src: mov(dest, src)},
    'MVI': {'opcode': 0x06, 'bytes': 2, 'operands': ['reg', 'data'], 'execute': lambda reg, data: mvi(reg, data)},
    'LXI': {'opcode': 0x01, 'bytes': 3, 'operands': ['rp', 'data16'],
            'execute': lambda rp, highByte, lowByte: lxi(rp, highByte, lowByte)},
    'LDA': {'opcode': 0xFA, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: lda(addr16)},
    'STA': {'opcode': 0xEA, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: sta(addr16)},
    'LHLD': {'opcode': 0x2A, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: lhld(addr16)},
    'SHLD': {'opcode': 0x22, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: shld(addr16)},
    'CPI': {'opcode': 0xFE, 'bytes': 2, 'operands': ['data'], 'execute': lambda data: cpi(data)},
    'RIM': {'opcode': 0x20, 'bytes': 1, 'operands': [], 'execute': lambda: rim()},
    'SIM': {'opcode': 0x30, 'bytes': 1, 'operands': [], 'execute': lambda: sim()},

    # Arithmetic Instructions
    'ADD': {'opcode': 0x80, 'bytes': 1, 'operands': ['reg'], 'execute': lambda reg: add(reg)},
    'ADI': {'opcode': 0xC6, 'bytes': 2, 'operands': ['data'], 'execute': lambda data: adi(data)},
    'SUB': {'opcode': 0x90, 'bytes': 1, 'operands': ['reg'], 'execute': lambda reg: sub(reg)},
    'SBI': {'opcode': 0xD6, 'bytes': 2, 'operands': ['data'], 'execute': lambda data: sbi(data)},
    'INR': {'opcode': 0x04, 'bytes': 1, 'operands': ['reg'], 'execute': lambda reg: incr(reg)},
    'DCR': {'opcode': 0x05, 'bytes': 1, 'operands': ['reg'], 'execute': lambda reg: dcr(reg)},

    # Logic Instructions
    'ANA': {'opcode': 0xA0, 'bytes': 1, 'operands': ['reg'], 'execute': lambda reg: ana(reg)},
    'ANI': {'opcode': 0xE6, 'bytes': 2, 'operands': ['data'], 'execute': lambda data: ani(data)},
    'XRA': {'opcode': 0xA8, 'bytes': 1, 'operands': ['reg'], 'execute': lambda reg: xra(reg)},
    'XRI': {'opcode': 0xEE, 'bytes': 2, 'operands': ['data'], 'execute': lambda data: xri(data)},
    'ORA': {'opcode': 0xB0, 'bytes': 1, 'operands': ['reg'], 'execute': lambda reg: ora(reg)},
    'ORI': {'opcode': 0xF6, 'bytes': 2, 'operands': ['data'], 'execute': lambda data: ori(data)},

    # Control Instructions
    'NOP': {'opcode': 0x00, 'bytes': 0, 'operands': [], 'execute': lambda: nop()},
    'HALT': {'opcode': 0x76, 'bytes': 1, 'operands': [], 'execute': lambda: halt()},
    'DI': {'opcode': 0xF3, 'bytes': 1, 'operands': [], 'execute': lambda: di()},
    'EI': {'opcode': 0xFB, 'bytes': 1, 'operands': [], 'execute': lambda: ei()},
    'RLC': {'opcode': 0x07, 'bytes': 1, 'operands': [], 'execute': lambda: rlc()},
    'RRC': {'opcode': 0x0F, 'bytes': 1, 'operands': [], 'execute': lambda: rrc()},
    'RAL': {'opcode': 0x17, 'bytes': 1, 'operands': [], 'execute': lambda: ral()},
    'RAR': {'opcode': 0x1F, 'bytes': 1, 'operands': [], 'execute': lambda: rar()},
    'CMA': {'opcode': 0x2F, 'bytes': 1, 'operands': [], 'execute': lambda: cma()},
    'CMC': {'opcode': 0x3F, 'bytes': 1, 'operands': [], 'execute': lambda: cmc()},
    'STC': {'opcode': 0x37, 'bytes': 1, 'operands': [], 'execute': lambda: stc()},
    'CMC': {'opcode': 0x3F, 'bytes': 1, 'operands': [], 'execute': lambda: cmc()},

    # Branch Instructions
    'JMP': {'opcode': 0xC3, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jmp(addr16)},
    'CALL': {'opcode': 0xCD, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: call(addr16)},
    'RET': {'opcode': 0xC9, 'bytes': 1, 'operands': [], 'execute': lambda: ret()},
    'RZ': {'opcode': 0xC8, 'bytes': 1, 'operands': [], 'execute': lambda: rz()},
    'RNZ': {'opcode': 0xC0, 'bytes': 1, 'operands': [], 'execute': lambda: rnz()},
    'RP': {'opcode': 0xF0, 'bytes': 1, 'operands': [], 'execute': lambda: rp()},
    'RM': {'opcode': 0xF8, 'bytes': 1, 'operands': [], 'execute': lambda: rm()},
    'RNC': {'opcode': 0xD0, 'bytes': 1, 'operands': [], 'execute': lambda: rnc()},
    'RC': {'opcode': 0xD8, 'bytes': 1, 'operands': [], 'execute': lambda: rc()},
    'JZ': {'opcode': 0xCA, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jz(addr16)},
    'JNZ': {'opcode': 0xC2, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jnz(addr16)},
    'JP': {'opcode': 0xE9, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jp(addr16)},
    'JM': {'opcode': 0xEA, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jm(addr16)},
    'JC': {'opcode': 0xDA, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jc(addr16)},
    'JNC': {'opcode': 0xD2, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jnc(addr16)},
    'JPO': {'opcode': 0xE3, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jpo(addr16)},
    'JPE': {'opcode': 0xE2, 'bytes': 3, 'operands': ['addr16'], 'execute': lambda addr16: jpe(addr16)},

    # Stack Instructions
    'PUSH': {'opcode': 0xC5, 'bytes': 1, 'operands': ['rp'], 'execute': lambda rp: push(rp)},
    'POP': {'opcode': 0xC1, 'bytes': 1, 'operands': ['rp'], 'execute': lambda rp: pop(rp)},
    'XTHL': {'opcode': 0xE3, 'bytes': 1, 'operands': [], 'execute': lambda: xthl()},
    'SPHL': {'opcode': 0xF9, 'bytes': 1, 'operands': [], 'execute': lambda: sp_hl()},

    # I/O Instructions
    'IN': {'opcode': 0xDB, 'bytes': 2, 'operands': ['port'], 'execute': lambda port: in_(port)},
    'OUT': {'opcode': 0xD3, 'bytes': 2, 'operands': ['port', 'data'], 'execute': lambda port, data: out(port, data)},
}


class Instruction:
    def __init__(self, mnemonic, operands, line_number):
        self.mnemonic = mnemonic
        self.operands = operands
        self.line_number = line_number
        self.machine_code = None

    def __repr__(self):
        return f"Instruction(mnemonic='{self.mnemonic}', operands={self.operands}, line={self.line_number}, machine_code={self.machine_code})"

    def execute(self):
        meta = instruction_metadata.get(self.mnemonic)
        if not meta:
            report_error(self.line_number, f"Unknown instruction: {self.mnemonic}")
            return

        execute_fn = meta['execute']
        execute_fn(*self.operands)


assembly_program = []
generated_machine_code = []
error_list = []


# File I/O Functions

@trace
def read_assembly_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        logger.error('Error: The file {file_path} was not found.')
        return None
    except IOError:
        logger.error('Error: Could not read the file {file_path}.')
        return None


@trace
def write_machine_code(file_path, machine_code):
    try:
        with open(file_path, 'wb') as file:
            file.write(bytes(machine_code))
    except IOError:
        logger.error('Error: Could not write to the file {file_path}.')


@trace
def write_error_log(file_path, error_list):
    try:
        with open(file_path, 'w') as file:
            for error in error_list:
                file.write(error + '\n')
    except IOError:
        logger.error('Error: Could not write to the file {file_path}.')


# Core Functions

@trace
def parse_line(line, line_number):
    parts = line.split()
    mnemonic = parts[0]
    operands = parts[1:] if len(parts) > 1 else []
    return Instruction(mnemonic, operands, line_number)


@trace
def parse_assembly_code(code):
    global assembly_program
    logger.info(f'\n\n**** Parsing assembly file ***')
    lines = code.splitlines()
    for i, line in enumerate(lines, start=1):
        if line.strip():
            try:
                instruction = parse_line(line.strip(), i)
                assembly_program.append(instruction)
            except Exception as e:
                report_error(i, str(e))


@trace
def report_error(line_number, message):
    error_list.append(f"Error on line {line_number}: {message}")


@trace
def generate_machine_code():
    global assembly_program, generated_machine_code, error_list
    logger.info(f'\n\n**** Generating Machine Code ***')

    generated_machine_code = []
    for instruction in assembly_program:
        meta = instruction_metadata.get(instruction.mnemonic)
        if not meta:
            report_error(instruction.line_number, f"Unknown instruction: {instruction.mnemonic}")
            continue

        opcode = meta['opcode']
        num_bytes = meta['bytes']
        operands = instruction.operands
        # Start with the opcode
        machine_code = [opcode]

        # Process operands based on the number of bytes and types specified in metadata
        if num_bytes > 1:
            for i, operand_type in enumerate(meta['operands']):
                if operand_type == 'reg':
                    # Register operand
                    if operands[i][0] not in register_code:
                        report_error(instruction.line_number, f"Unknown register: {operands[i]}")
                        break
                    machine_code.append(register_code[operands[i][0]])
                elif operand_type == 'data':
                    # Immediate data operand
                    try:
                        # Handle hexadecimal data
                        data = int(operands[i], 16)
                        machine_code.append(data)
                    except ValueError:
                        report_error(instruction.line_number, f"Invalid data value: {operands[i]}")
                        break
                elif operand_type == 'data16':
                    # 16-bit address operand
                    try:
                        # Handle hexadecimal address
                        address = int(operands[i], 16)
                        machine_code.append(address & 0xFF)  # Low byte
                        machine_code.append((address >> 8) & 0xFF)  # High byte
                    except ValueError:
                        report_error(instruction.line_number, f"Invalid address value: {operands[i]}")
                        break
                elif operand_type == 'rp':
                    # Register pair operand (not explicitly defined in your case but can be added if needed)
                    if operands[i][0] not in register_code:
                        report_error(instruction.line_number, f"Unknown register: {operands[i]}")
                    elif operands[i][0] != 'B' and operands[i][0] != 'D' and operands[i][0] != 'H':
                        report_error(instruction.line_number, f"Invalid register pair: {operands[i]}")
                    machine_code.append(register_code[operands[i][0]])
                else:
                    report_error(instruction.line_number, f"Unknown operand type: {operand_type}")
                    break

        # Append the generated machine code for this instruction to the list
        generated_machine_code.extend(machine_code)


# Instruction execution functions

@trace
def load_program_from_bin(file_path, start_address):
    """
    Loads a binary program from a .bin file into memory at the specified address.

    Args:
        file_path (str): The path to the .bin file containing the program.
        start_address (int): The memory address where the program should be loaded.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the start address is out of memory bounds.
    """
    global memory
    logger.info(f'\n\n**** Loading Program to Memory ***')

    # Check if the start address is within the valid memory range.
    if start_address < 0 or start_address >= len(memory):
        raise ValueError("Start address is out of memory bounds.")

    # Read binary data from file.
    try:
        with open(file_path, 'rb') as file:
            # Load the binary data into memory.
            binary_data = file.read()

            # Ensure there is enough space in memory to load the binary data.
            if start_address + len(binary_data) > len(memory):
                raise ValueError("Binary data exceeds available memory space.")

            # Copy the binary data to memory.
            memory[start_address:start_address + len(binary_data)] = binary_data

    except FileNotFoundError:
        logger.error("Error: The file '{file_path}' does not exist.")
        raise
    except Exception as e:
        logger.error('Error loading binary file: {e}')
        raise

    logger.info(f"Program loaded from '{file_path}' into memory starting at address {start_address:04X}.")


@trace
def execute_program(start_address):
    """
    Executes a program starting from the given address.

    Args:
        start_address (int): The starting address of the program in memory.
    """

    global registers, memory, flags, stack_pointer, program_counter
    logger.info(f'\n\n**** Executing Program ***')

    # Set the program counter to the start address.
    program_counter = start_address

    while True:
        # Check if the program counter is within valid memory bounds.
        if program_counter < 0 or program_counter >= len(memory):
            logger.error('Program counter {program_counter:04X} is out of bounds.')
            break

        # Fetch the instruction opcode from memory.
        opcode = memory[program_counter]

        # Check if the opcode is valid.
        if opcode not in opcode_instruction_map:
            logger.error('Invalid opcode {opcode:02X} at address {program_counter:04X}')
            break

        instruction = opcode_instruction_map[opcode]
        instr_metadata = instruction_metadata.get(instruction)

        if not instr_metadata:
            logger.error("Instruction metadata for '{instruction}' not found.")
            break

        num_bytes = instr_metadata['bytes']

        logger.info(f'opcode = {opcode}, iMetadata = {instr_metadata}, pc = {program_counter}')

        # Fetch the operands (if any) from memory.
        operand_location = program_counter + 1
        operands = []
        for i in range(len(instr_metadata["operands"])):
            operand_type = instr_metadata["operands"][i]

            if operand_type == "reg" or operand_type == "rp":
                if operand_location + i >= len(memory):
                    logger.error('Operand location {operand_location + i} is out of bounds.')
                    break
                reg_code = memory[operand_location + i]
                reg_name = code_register.get(reg_code)
                if not reg_name:
                    logger.error('Unknown register code {reg_code:02X}.')
                    break
                operands.append(reg_name)

            elif operand_type == "data":
                if operand_location + i >= len(memory):
                    logger.error('Operand location {operand_location + i} is out of bounds.')
                    break
                operands.append(memory[operand_location + i])
            elif operand_type == "data16":
                operands.append(memory[operand_location + i])
                operands.append(memory[operand_location + i + 1])

        logger.info(f'operands = {operands}')

        # Increment the program counter.
        program_counter += (num_bytes + 1)

        # Execute the instruction.
        if 'execute' in instr_metadata:
            try:
                instr_metadata['execute'](*operands)
            except Exception as e:
                logger.error('Error executing instruction at address {program_counter - (num_bytes + 1):04X}: {e}')
                break

        # Handle HALT instruction
        if opcode == 0x76:  # HALT opcode
            logger.error('HALT instruction encountered.')
            break
        elif opcode == 0x00:
            break;
    logger.info(f'Program execution completed.')

    # Registers
    headers = list(registers.keys())
    values = list(registers.values())
    print(tabulate([values], headers=headers, tablefmt="grid"))

    # Flags
    headers = list(flags.keys())
    values = list(flags.values())
    print(tabulate([values], headers=headers, tablefmt="grid"))

    # Memory
    chunk_size = 16
    headers = [f"0x{(i * chunk_size):04X}" for i in range(len(memory) // chunk_size)]
    rows = [memory[i:i + chunk_size] for i in range(0, len(memory), chunk_size)]
    # print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Stack and program counter
    headers = ["Register", "Value"]
    data = [
        ["Stack Pointer (SP)", hex(stack_pointer)],
        ["Program Counter (PC)", hex(program_counter)]
    ]
    print(tabulate(data, headers=headers, tablefmt="grid"))


# Complete process
@trace
def assemble_and_execute(input_file, output_file, error_log_file):
    # Step 1: Read the assembly source code
    source_code = read_assembly_file(input_file)
    if source_code is None:
        return
    # Step 2: Parse the assembly code
    parse_assembly_code(source_code)

    # Step 3: Generate machine code
    generate_machine_code()
    # Step 4: Execute the program
    if not error_list:
        write_machine_code(output_file, generated_machine_code)
        load_program_from_bin(output_file, 0)
        print(memory[:11])
        execute_program(0)
        print(f"Execution completed.")
    else:
        # If there are errors, write them to the error log
        write_error_log(error_log_file, error_list)
        print(f"Errors found. See {error_log_file} for details.")


# Example usage
input_file = "program.asm"
output_file = "program.bin"
error_log_file = "errors.log"

assemble_and_execute(input_file, output_file, error_log_file)


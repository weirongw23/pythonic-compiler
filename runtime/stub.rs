use std::collections::HashSet;

#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone)]
struct SnakeVal(u64);

#[repr(C)]
struct SnakeArray {
    size: u64,
    elts: *const SnakeVal,
}

#[repr(u64)]
pub enum SnakeErr {
    Overflow = 0,
    ArithExpectedNum = 1,
    CmpExpectedNum = 2,
    LogExpectedBool = 3,
    IfExpectedBool = 4,
    IndexOutOfBounds = 5,
    IndexedNonArray = 6,
    IndexNotNum = 7,
    LengthNonArray = 8,
    ValueUsedAsFunction = 9,
    FunctionCalledWrongArity = 10
}

static BOOL_TAG: u64 = 0x00_00_00_00_00_00_00_01;
static SNAKE_TRU: SnakeVal = SnakeVal(0xFF_FF_FF_FF_FF_FF_FF_FF);
static SNAKE_FLS: SnakeVal = SnakeVal(0x7F_FF_FF_FF_FF_FF_FF_FF);
static SNAKE_PADDING: SnakeVal = SnakeVal(0x01_00_FF_FF_FF_FF_FF_FF);

#[link(name = "compiled_code", kind = "static")]
extern "sysv64" {

    // The \x01 here is an undocumented feature of LLVM that ensures
    // it does not add an underscore in front of the name.
    #[link_name = "\x01start_here"]
    fn start_here() -> SnakeVal;
}

// reinterprets the bytes of an unsigned number to a signed number
fn unsigned_to_signed(x: u64) -> i64 {
    i64::from_le_bytes(x.to_le_bytes())
}

/* You can use this function to cast a pointer to an array on the heap
 * into something more convenient to access
 *
 */
fn load_snake_array(p: *const u64) -> SnakeArray {
    unsafe {
        let size = *p;
        SnakeArray {
            size,
            elts: std::mem::transmute(p.add(1)),
        }
    }
}

fn sprint_array(array: &SnakeArray, seen: &mut HashSet<u64>) -> String {
    let mut is = String::from("[");
    for i in 0..array.size {
        unsafe {
            let val = array.elts.add(i as usize);
            is.push_str(&sprint_snake_val(*val, seen));
        }

        if i != array.size - 1 {
            is.push_str(", ");
        }
    }
    is.push(']');
    is
}

fn sprint_snake_val(x: SnakeVal, seen: &mut HashSet<u64>) -> String {
    let result = x.0 & 0x00_00_00_00_00_00_00_07;
    if x.0 & BOOL_TAG == 0 {
        format!("{}", unsigned_to_signed(x.0) >> 1)
    } else if x == SNAKE_TRU {
        String::from("true")
    } else if x == SNAKE_FLS {
        String::from("false")
    } else if result == 3 {
        format!("<closure>")
    } else if result == 1 {
        let array = load_snake_array((x.0 - 0x1) as *const u64);
        if seen.contains(&x.0) {
            return String::from("<loop>");
        }
        seen.insert(x.0);
        let result = sprint_array(&array, seen);
        seen.remove(&x.0);
        return result;
    } else {
        format!("Invalid snake value 0x{:x}", x.0)
    }
}

#[export_name = "\x01err"]
extern "sysv64" fn err() {
    std::process::exit(1)
}

#[export_name = "\x01print_snake_val"]
extern "sysv64" fn print_snake_val(v: SnakeVal) -> SnakeVal {
    println!("{}", sprint_snake_val(v, &mut HashSet::new()));
    v
}

#[export_name = "\x01dbg"]
extern "sysv64" fn dbg(stack: u64) {
    println!("Rsp: 0x{:x}", stack);
}

#[export_name = "\x01snake_error"]
extern "sysv64" fn snake_error(ecode: SnakeErr, v1: SnakeVal) -> SnakeVal {
    match ecode {
        SnakeErr::Overflow => eprintln!("Operation overflowed"),
        SnakeErr::ArithExpectedNum => {
            eprintln!("arithmetic expected a number")
        }
        SnakeErr::CmpExpectedNum => {
            eprintln!("comparison expected a number")
        }
        SnakeErr::LogExpectedBool => {
            eprintln!("logic expected a boolean")
        }
        SnakeErr::IfExpectedBool => {
            eprintln!("if expected a boolean")
        }
        SnakeErr::IndexOutOfBounds => {
            eprintln!("index out of bounds")
        }
        SnakeErr::IndexedNonArray => {
            eprintln!("indexed into non-array")
        }
        SnakeErr::IndexNotNum => {
            eprintln!("index not a number")
        }
        SnakeErr::LengthNonArray => {
            eprintln!("length called with non-array")
        }
        SnakeErr::ValueUsedAsFunction => {
            eprintln!("called a non-function")
        }
        SnakeErr::FunctionCalledWrongArity => {
            eprintln!("wrong number of arguments")
        }
    }
    std::process::exit(1)
}

fn main() {
    let output = unsafe { start_here() };
    println!("{}", sprint_snake_val(output, &mut HashSet::new()));
}

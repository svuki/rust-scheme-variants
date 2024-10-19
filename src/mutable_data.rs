use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// SExpressions are composed of atoms and lists.
// For simplicity, we'll represent atoms as strings and
// lists as vectors in rust.
enum SExpression {
    Atom(String),
    List(Vec<SExpression>),
}

// read takes in a string and returns an SExpression
fn read(s: String) -> SExpression {
    Reader::read(s)
}

// This class handles converting string input to an SExpression.
struct Reader {
    tokens: Vec<String>,
    index: usize,
}

impl Reader {
    pub fn read(s: String) -> SExpression {
        let mut reader = Self::of_string(s);
        reader.read_sexpr()
    }

    fn of_string(s: String) -> Self {
        // Quick and dirty implementation. We insert spaces
        // around opening and close parentheses so we can then
        // split on whitespace and get open/close parens as distinct
        // tokens.
        let tokens: Vec<String> = s
            .replace("(", " ( ")
            .replace(")", " ) ")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        for t in &tokens {
            println!("{}", t);
        }
        Self { tokens, index: 0 }
    }

    fn read_sexpr(&mut self) -> SExpression {
        if self.peek() == "(" {
            self.read_list()
        } else {
            self.read_atom()
        }
    }

    fn read_list(&mut self) -> SExpression {
        assert_eq!(self.take(), "(");
        let mut values = vec![];
        loop {
            let token = self.peek();
            if token == ")" {
                self.take();
                break;
            };
            values.push(self.read_sexpr());
        }
        SExpression::List(values)
    }

    fn read_atom(&mut self) -> SExpression {
        SExpression::Atom(self.take().to_string())
    }

    fn take(&mut self) -> &str {
        let x = &self.tokens[self.index];
        self.index += 1;
        x
    }

    fn peek(&self) -> &str {
        &self.tokens[self.index]
    }
}

// For this sketch, we'll just consider a limited set of values,
// enough to implement a factorial function
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Nil,         // nil is the only falsey value, we use it in conditional tests
    Number(f64), // for simplicity, just use floats for all numbers
    Symbol(String),
    Procedure(Proc), // procedures are first class values
    PrimitiveProc(fn(Vec<ValueRef>) -> ValueRef),
    Cons(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
}

type ValueRef = Rc<RefCell<Value>>;

impl Value {
    fn to_ref(self) -> ValueRef {
        Rc::new(RefCell::new(self))
    }
}

fn expect_cons(v: ValueRef) -> (Rc<RefCell<Value>>, Rc<RefCell<Value>>) {
    println!("EXPECT_CONS: {:#?}", v);
    match &*v.borrow() {
        Value::Cons(x, y) => (x.clone(), y.clone()),
        _ => panic!("Expecting a cons cell."),
    }
}

fn cons_to_vector(v: ValueRef) -> Vec<ValueRef> {
    fn cons_to_vector_iter(v: ValueRef, mut vec: Vec<ValueRef>) -> Vec<ValueRef> {
        match &*v.borrow() {
            Value::Nil => vec,
            Value::Cons(head, tail) => {
                vec.push(head.clone());
                cons_to_vector_iter(tail.clone(), vec)
            }
            _ => panic!("Expecting a cons cell."),
        }
    }
    let vec = Vec::new();
    return cons_to_vector_iter(v, vec);
}

fn vector_to_cons(v: Vec<ValueRef>) -> ValueRef {
    let mut last = Rc::new(RefCell::new(Value::Nil));
    for val in v.into_iter().rev() {
        last = Rc::new(RefCell::new(Value::Cons(val, last)))
    }
    last
}

#[derive(Clone, Debug, PartialEq)]
struct Proc {
    arglist: Vec<String>,
    body: ValueRef,
    env: Rc<RefCell<Env>>,
}

// Parse takes in the SExpression and converts it to a Value.
// This takes care of properly reading things like numbers, symbols, etc.
// This could be a part of the read function above, but it is cleaner to
// separate it. The alternative would be to have `read` above handle this parsing
// and to remove the explicit SExpression type.
fn parse(sexpr: SExpression) -> ValueRef {
    match sexpr {
        SExpression::List(sexprs) => {
            let child_sexprs = sexprs.into_iter().map(|v| parse(v)).collect();
            vector_to_cons(child_sexprs)
        }
        SExpression::Atom(s) => match s.parse::<f64>() {
            Ok(number) => Rc::new(RefCell::new(Value::Number(number))),
            Err(_) => Rc::new(RefCell::new(Value::Symbol(s.to_string()))),
        },
    }
}

// Environments hold mappings from symbols to values. They contain a pointer
// to their parent environment so lookups can be forwarded to the enclosing
// environment in case the current environment doesn't have a match.
#[derive(Clone, Debug, PartialEq)]
struct Env {
    bindings: HashMap<String, ValueRef>,
    parent: Option<Rc<RefCell<Env>>>,
}

fn new_child_env(parent: Rc<RefCell<Env>>) -> Rc<RefCell<Env>> {
    let mut env = Env::new();
    env.parent = Some(parent.clone());
    Rc::new(RefCell::new(env))
}

impl Env {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            parent: None,
        }
    }

    // Need to include some primitive procedures to actually make anything
    // useful.
    fn new_with_primitives() -> Self {
        let mut env = Self::new();
        env.bind(
            "-".to_string(),
            Value::PrimitiveProc(primitive_minus).to_ref(),
        );
        env.bind(
            "*".to_string(),
            Value::PrimitiveProc(primitive_multiply).to_ref(),
        );
        env.bind("=".to_string(), Value::PrimitiveProc(primitive_eq).to_ref());
        env.bind(
            "+".to_string(),
            Value::PrimitiveProc(primitive_plus).to_ref(),
        );
        env
    }

    fn lookup(&self, name: String) -> ValueRef {
        println!("looking up {name}");
        match self.bindings.get(&name) {
            Some(value) => value.clone(),
            None => match &self.parent {
                Some(parent) => parent.borrow().lookup(name),
                None => panic!("Could not look up variable."),
            },
        }
    }

    fn bind(&mut self, name: String, value: ValueRef) {
        self.bindings.insert(name, value);
    }

    fn bind_many(&mut self, names: Vec<String>, values: Vec<ValueRef>) {
        for (name, val) in names.into_iter().zip(values.into_iter()) {
            self.bind(name, val)
        }
    }

    fn set(&mut self, name: String, value: ValueRef) {
        if self.bindings.contains_key(&name) {
            self.bind(name, value)
        } else {
            match &mut self.parent {
                Some(parent) => parent.borrow_mut().set(name, value),
                None => panic!("Could not find variable in any enclosing environment."),
            }
        }
    }

    fn show(&self, prefix: &str) {
        for key in self.bindings.keys() {
            println!("{}{}", prefix, key);
        }
        match &self.parent {
            Some(parent) => parent.borrow().show(&(prefix.to_string() + "  ")),
            _ => return,
        }
    }
}

// Eval implements the evaluation rules for our interpreter.
fn eval(env: Rc<RefCell<Env>>, expr: ValueRef) -> ValueRef {
    println!("Evaluate: {:#?}", expr);
    env.borrow().show("");
    match &*expr.borrow() {
        Value::Number(_) => expr.clone(), // numbers evaluate to themselves
        Value::Symbol(s) => {
            if s == "'nil" {
                Value::Nil.to_ref()
            } else {
                // find the bound value for the symbol
                let value = env.borrow().lookup(s.to_string());
                println!("VALUE: {:#?}", value);
                value
            }
        }
        // There are a limited number of primitive "special" forms that have special
        // evaluation rules. These are primitive operations for conditionals, environment
        // manipulation, and procedure creation.
        Value::Cons(car, cdr) if is_special(expect_symbol(&*car.borrow())) => {
            println!("Applying special: {:#?}", car);
            let r = apply_special(env, expr.clone());
            println!("OK, done applying special {:#?}", car);
            r
        }
        Value::Cons(_, _) => apply(env, expr.clone()),
        Value::Nil => panic!("Error, tried to evaluate nil."),
        Value::Procedure(_) => panic!("Tried to evaluate proc."),
        Value::PrimitiveProc(_) => panic!("Trued to evaluate a primitive proc."),
    }
}

const SPECIAL_DEFINE: &str = "define";
const SPECIAL_IF: &str = "if";
const SPECIAL_PROGN: &str = "progn";
const SPECIAL_LAMBDA: &str = "lambda";
const SPECIAL_SET: &str = "set!";
const SPECIAL_CONS: &str = "cons";
const SPECIAL_CAR: &str = "car";
const SPECIAL_SET_CAR: &str = "set-car!";
const SPECIAL_CDR: &str = "cdr";
const SPECIAL_SET_CDR: &str = "set-cdr!";

const SPECIAL_SYMBOLS: [&str; 10] = [
    SPECIAL_DEFINE,
    SPECIAL_IF,
    SPECIAL_PROGN,
    SPECIAL_LAMBDA,
    SPECIAL_SET,
    SPECIAL_CONS,
    SPECIAL_CAR,
    SPECIAL_CDR,
    SPECIAL_SET_CAR,
    SPECIAL_SET_CDR,
];

// We need special handling wherever the default "apply" handling
// of evaluating all the arguments doesn't work.
fn apply_special(env: Rc<RefCell<Env>>, contents: ValueRef) -> ValueRef {
    let mut c_iter = cons_to_vector(contents).into_iter();
    let head = c_iter.next().unwrap();
    if let Value::Symbol(ref s) = &*head.clone().borrow() {
        match s.as_ref() {
            SPECIAL_DEFINE => {
                // For define, we bind the second element of the form to
                // the evaluated value of the third element.
                if let Value::Symbol(ref varname) = &*c_iter.next().unwrap().borrow() {
                    let expr = c_iter.next().unwrap();
                    let value = eval(env.clone(), expr);
                    env.borrow_mut().bind(varname.clone(), value);
                    Value::Nil.to_ref()
                } else {
                    panic!("Tried to bind a non symbol.")
                }
            }
            SPECIAL_IF => {
                // If triggers evaluation of the second or third element depending
                // on if the condition is true or false. Only the nil value is treated
                // as false, anything else is treated as true.
                let condition = c_iter.next().unwrap();
                let true_case = c_iter.next().unwrap();
                let false_case = c_iter.next().unwrap();
                match &*eval(env.clone(), condition).borrow() {
                    Value::Nil => eval(env, false_case),
                    _ => eval(env, true_case),
                }
            }
            SPECIAL_PROGN => {
                // progn evaluates a sequence of expressions and returns the value
                // of the last one. Each expression is evaluated in the same environment,
                // so if any of them changes the environment, subsequent evaluations
                // will see that change.
                let mut last = Value::Nil.to_ref();
                let progn_env = new_child_env(env);
                for v in c_iter {
                    last = eval(progn_env.clone(), v.clone())
                }
                last
            }
            SPECIAL_LAMBDA => {
                // lambda creates a procedure
                let args = cons_to_vector(c_iter.next().unwrap());
                let arglist: Vec<String> = args
                    .iter()
                    .map(|x| expect_symbol(&*x.borrow()).to_string())
                    .collect();
                let body = c_iter.next().unwrap();
                Value::Procedure(Proc {
                    arglist,
                    body,
                    env: env.clone(),
                })
                .to_ref()
            }
            SPECIAL_SET => {
                let variable = c_iter.next().unwrap();
                let value = eval(env.clone(), c_iter.next().unwrap());
                if let Value::Symbol(ref name) = &*variable.borrow() {
                    env.borrow_mut().set(name.clone(), value);
                } else {
                    panic!("Tried to a set a nonsymbol.")
                }
                Value::Nil.to_ref()
            }
            SPECIAL_CONS => {
                let car = eval(env.clone(), c_iter.next().unwrap());
                let cdr = eval(env.clone(), c_iter.next().unwrap());
                Value::Cons(car, cdr).to_ref()
            }
            SPECIAL_CAR => {
                let v = c_iter.next().unwrap();
                println!("CAR OF: {:#?}", v.clone());
                let (car, _) = expect_cons(eval(env.clone(), v.clone()));
                println!("CAR DONE");
                car
            }
            SPECIAL_CDR => {
                let (_, cdr) = expect_cons(eval(env.clone(), c_iter.next().unwrap()));
                cdr
            }
            SPECIAL_SET_CAR => {
                let cons = eval(env.clone(), c_iter.next().unwrap());
                let (_, cdr) = expect_cons(cons.clone());
                let value = eval(env.clone(), c_iter.next().unwrap());
                cons.replace(Value::Cons(value, cdr));
                Value::Nil.to_ref()
            }
            SPECIAL_SET_CDR => {
                let cons = eval(env.clone(), c_iter.next().unwrap());
                let (car, _) = expect_cons(cons.clone());
                let value = eval(env.clone(), c_iter.next().unwrap());
                cons.replace(Value::Cons(car, value));
                Value::Nil.to_ref()
            }
            _ => panic!("Unknown special form."),
        }
    } else {
        panic!("Tried to apply special form but first element was not a symbol.")
    }
}

fn is_special(name: &str) -> bool {
    SPECIAL_SYMBOLS.contains(&name)
}

// Apply implements the default procedure application rule.
fn apply(env: Rc<RefCell<Env>>, values: ValueRef) -> ValueRef {
    let values = cons_to_vector(values);
    // first evaluate the head
    match &*eval(env.clone(), values[0].clone()).borrow() {
        Value::Procedure(proc) => {
            // If it is a procedure, then evalaute all the arguments
            let args = values[1..]
                .iter()
                .map(|v| eval(env.clone(), v.clone()))
                .collect();
            // Construct a child environment of the current environment and bind
            // the arguments to the procedure's free variables (ie, the argument list).
            let apply_env = new_child_env(proc.env.clone());
            apply_env.borrow_mut().bind_many(proc.arglist.clone(), args);
            // And evaluate the body of the procedure in the resulting environment
            eval(apply_env, proc.body.clone())
        }
        Value::PrimitiveProc(native_func) => {
            // If it is a primitive procedure, valuate the arguments and pass them
            // directly to the rust function implementing the procedure.
            let args: Vec<ValueRef> = values[1..]
                .iter()
                .map(|v| eval(env.clone(), v.clone()))
                .collect();
            return native_func(args);
        }
        _ => panic!("Tried to apply a non procedure."),
    }
}

//the show function converts a value to a string for printing
fn show(value: Value) -> String {
    // to speed things up we'll use the default debug formatting
    format!("{:#?}", value)
}

fn expect_symbol(v: &Value) -> &str {
    if let Value::Symbol(s) = v {
        &s
    } else {
        panic!("Expecting a symbol.")
    }
}

fn expect_number(v: ValueRef) -> f64 {
    if let Value::Number(x) = &*v.borrow() {
        x.clone()
    } else {
        panic!("Expecting a number.")
    }
}

fn primitive_minus(values: Vec<ValueRef>) -> ValueRef {
    let mut count = expect_number(values[0].clone());
    for v in values[1..].iter() {
        count -= expect_number(v.clone())
    }
    Value::Number(count).to_ref()
}

fn primitive_multiply(values: Vec<ValueRef>) -> ValueRef {
    let mut count = expect_number(values[0].clone());
    for v in &values[1..] {
        count *= expect_number(v.clone())
    }
    Value::Number(count).to_ref()
}

fn primitive_plus(values: Vec<ValueRef>) -> ValueRef {
    let mut count = 0.;
    for v in &values {
        count += expect_number(v.clone())
    }
    Value::Number(count).to_ref()
}

fn primitive_eq(values: Vec<ValueRef>) -> ValueRef {
    let fst = values[0].clone();
    for v in &values[1..] {
        if !(&*fst.borrow() == &*v.borrow()) {
            return Value::Nil.to_ref();
        }
    }
    Value::Symbol("true".to_string()).to_ref()
}

#[cfg(test)]
mod test {
    use super::*;

    // #[test]
    fn test_set() {
        let program = "
(progn
  (define x 3)
  (set! x 100)
  x)
";
        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(100.).to_ref());
    }

    // #[test]
    fn test_closure() {
        let program = "
(progn
  (define make-counter
     (lambda ()
        (progn
          (define counter 0)
          (lambda (value)
            (if (= 0 value)
              counter
              (set! counter (+ counter value)))))))
  (define counter (make-counter))
  (counter 100)
  (counter 101)
  (counter 0))
";
        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(201.).to_ref());
    }

    #[test]
    fn test_mutable_data1() {
        let program = "
(progn
  (define my-list (cons 1 (cons 2 (cons 3 'nil))))
  (car my-list))
";
        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(1.).to_ref());
    }

    #[test]
    fn test_mutable_data2() {
        let program = "
(progn
  (define my-list (cons 1 (cons 2 (cons 3 'nil))))
  (define my-other-list (cdr my-list))
  (car my-other-list))
";
        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(2.).to_ref());
    }

    #[test]
    fn test_mutable_data3() {
        let program = "
(progn
  (define my-list (cons 1 (cons 2 (cons 3 'nil))))
  (set-car! my-list 10)
  (car my-list))
";

        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(10.).to_ref());
    }

    #[test]
    fn test_mutable_data4() {
        let program = "
(progn
  (define my-list (cons 1 (cons 2 (cons 3 'nil))))
  (set-cdr! my-list (cons 5 (cons 10 'nil)))
  (car my-list))
";

        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(1.).to_ref());
    }

    #[test]
    fn test_mutable_data5() {
        let program = "
(progn
  (define my-list (cons 1 (cons 2 (cons 3 'nil))))
  (set-cdr! my-list (cons 5 (cons 10 'nil)))
  (car (cdr my-list)))
";

        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(5.).to_ref());
    }

    #[test]
    fn test_mutable_data6() {
        let program = "
(progn
  (define my-list (cons 1 (cons 2 (cons 3 'nil))))
  (define my-other-list (cdr my-list))
  (set-car! my-other-list 20)
  (car (cdr my-list)))
";

        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let root_env = Rc::new(RefCell::new(Env::new_with_primitives()));
        let result = eval(root_env, expr);
        assert_eq!(result, Value::Number(20.).to_ref());
    }
}

use std::collections::HashMap;

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
// enough to impement a factorial function
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Nil,         // nil is the only falsey value, we use it in conditional tests
    Number(f64), // for simplicity, just use floats for all numbers
    Symbol(String),
    Procedure(Proc), // procedures are first class values
    PrimitiveProc(fn(Vec<Value>) -> Value),
    List(Vec<Value>),
}

#[derive(Clone, Debug, PartialEq)]
struct Proc {
    arglist: Vec<String>,
    body: Box<Value>,
}

// Parse takes in the SExpression and converts it to a Value.
// This takes care of properly reading things like numbers, symbols, etc.
// This could be a part of the read function above, but it is cleaner to
// separate it. The alternative would be to have `read` above handle this parsing
// and to remove the explicit SExpression type.
fn parse(sexpr: SExpression) -> Value {
    match sexpr {
        SExpression::List(sexprs) => Value::List(sexprs.into_iter().map(|v| parse(v)).collect()),
        SExpression::Atom(s) => match s.parse::<f64>() {
            Ok(number) => Value::Number(number),
            Err(_) => Value::Symbol(s.to_string()),
        },
    }
}

// Environments hold mappings from symbols to values. They contain a pointer
// to their parent environment so lookups can be forwarded to the enclosing
// environment in case the current environment doesn't have a match.
struct Env<'a> {
    bindings: HashMap<String, Value>,
    parent: Option<&'a Env<'a>>,
}

impl<'a> Env<'a> {
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
        env.bind("-".to_string(), Value::PrimitiveProc(primitive_minus));
        env.bind("*".to_string(), Value::PrimitiveProc(primitive_multiply));
        env.bind("=".to_string(), Value::PrimitiveProc(primitive_eq));
        env
    }

    fn new_child_env(&'a self) -> Self {
        Self {
            bindings: HashMap::new(),
            parent: Some(&self),
        }
    }

    fn lookup(&'a self, name: String) -> Value {
        println!("looking up {name}");
        match self.bindings.get(&name) {
            Some(value) => value.clone(),
            None => match self.parent {
                Some(parent) => parent.lookup(name),
                None => panic!("Could not look up variable."),
            },
        }
    }

    fn bind(&mut self, name: String, value: Value) {
        self.bindings.insert(name, value);
    }

    fn bind_many(&mut self, names: Vec<String>, values: Vec<Value>) {
        for (name, val) in names.into_iter().zip(values.into_iter()) {
            self.bind(name, val)
        }
    }
}

// Eval implements the evaluation rules for our interpreter.
fn eval(env: &mut Env, expr: Value) -> Value {
    match expr {
        Value::Number(_) => expr,          // numbers evaluate to themselves
        Value::Symbol(s) => env.lookup(s), // find the bound value for the symbol
        // There are a limited number of primitive "special" forms that have special
        // evaluation rules. These are primitive operations for conditionals, environment
        // manipulation, and procedure creation.
        Value::List(contents) if is_special(contents[0].clone()) => {
            println!("Applying special: {:#?}", &contents[0]);
            apply_special(env, contents)
        }
        Value::List(contents) => apply(env, contents),
        Value::Nil => panic!("Error, tried to evaluate nil."),
        Value::Procedure(_) => panic!("Tried to evaluate proc."),
        Value::PrimitiveProc(_) => panic!("Trued to evaluate a primitive proc."),
    }
}

const SPECIAL_DEFINE: &str = "define";
const SPECIAL_IF: &str = "if";
const SPECIAL_PROGN: &str = "progn";
const SPECIAL_LAMBDA: &str = "lambda";
const SPECIAL_SYMBOLS: [&str; 4] = [SPECIAL_DEFINE, SPECIAL_IF, SPECIAL_PROGN, SPECIAL_LAMBDA];

// We need special handling wherever the default "apply" handling
// of evaluating all the arguments doesn't work.
fn apply_special(env: &mut Env, contents: Vec<Value>) -> Value {
    if let Value::Symbol(ref s) = contents[0] {
        match s.as_ref() {
            SPECIAL_DEFINE => {
                // For define, we bind the second element of the form to
                // the evaluated value of the third element.
                if let Value::Symbol(ref varname) = contents[1] {
                    let value = eval(env, contents[2].clone());
                    env.bind(varname.clone(), value);
                    Value::Nil
                } else {
                    panic!("Tried to bind a non symbol.")
                }
            }
            SPECIAL_IF => match eval(env, contents[1].clone()) {
                // If triggers evaluation of the second or third element depending
                // on if the condition is true or false. Only the nil value is treated
                // as false, anything else is treated as true.
                Value::Nil => eval(env, contents[3].clone()),
                _ => eval(env, contents[2].clone()),
            },
            SPECIAL_PROGN => {
                // progn evaluates a sequence of expressions and returns the value
                // of the last one. Each expression is evaluated in the same environment,
                // so if any of them changes the environment, subsequent evaluations
                // will see that change.
                let mut last = Value::Nil;
                for v in contents[1..].iter() {
                    last = eval(env, v.clone())
                }
                last
            }
            SPECIAL_LAMBDA => {
                // lambda creates a procedure
                let arglist: Vec<String> = match &contents[1] {
                    Value::List(contents) => contents
                        .iter()
                        .map(|x| expect_symbol(x).to_string())
                        .collect(),
                    _ => panic!("Malformed arglist."),
                };
                let body = &contents[2];
                Value::Procedure(Proc {
                    arglist,
                    body: Box::new(body.clone()),
                })
            }
            _ => panic!("Unknown special form."),
        }
    } else {
        panic!("Tried to apply special form but first element was not a symbol.")
    }
}

fn is_special(expr: Value) -> bool {
    if let Value::Symbol(ref s) = expr {
        // here: compiler made me do this...
        SPECIAL_SYMBOLS.contains(&s.as_ref())
    } else {
        false
    }
}

// Apply implements the default procedure application rule.
fn apply(env: &mut Env, values: Vec<Value>) -> Value {
    // first evaluate the head
    match eval(env, values[0].clone()) {
        Value::Procedure(proc) => {
            // If it is a procedure, then evalaute all the arguments
            let args = values[1..].iter().map(|v| eval(env, v.clone())).collect();
            // Construct a child environment of the current environment and bind
            // the arguments to the procedure's free variables (ie, the argument list).
            let mut apply_env = env.new_child_env();
            apply_env.bind_many(proc.arglist, args);
            // And evaluate the body of the procedure in the resulting environment
            eval(&mut apply_env, *proc.body)
        }
        Value::PrimitiveProc(native_func) => {
            // If it is a primitive procedure, valuate the arguments and pass them
            // directly to the rust function implementing the procedure.
            let args = values[1..].iter().map(|v| eval(env, v.clone())).collect();
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

fn expect_number(v: &Value) -> f64 {
    if let Value::Number(x) = v {
        x.clone()
    } else {
        panic!("Expecting a number.")
    }
}

fn primitive_minus(values: Vec<Value>) -> Value {
    let mut count = expect_number(&values[0]);
    for v in &values[1..] {
        count -= expect_number(v)
    }
    Value::Number(count)
}

fn primitive_multiply(values: Vec<Value>) -> Value {
    let mut count = expect_number(&values[0]);
    for v in &values[1..] {
        count *= expect_number(v)
    }
    Value::Number(count)
}

fn primitive_eq(values: Vec<Value>) -> Value {
    let fst = &values[0];
    for v in &values[1..] {
        if !(fst == v) {
            return Value::Nil;
        }
    }
    Value::Symbol("true".to_string())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_case() {
        let program = "
(progn
  (define factorial
    (lambda (n)
      (if (= n 0)
  	1
     	(* n (factorial (- n 1))))))

  (factorial 5))
";
        let expr = parse(read(program.to_string()));
        println!("Expr: {:#?}", expr);
        let mut root_env = Env::new_with_primitives();
        let result = eval(&mut root_env, expr);
        assert_eq!(result, Value::Number(120.));
    }
}

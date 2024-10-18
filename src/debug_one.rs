struct Node<'a> {
    val: i32,
    parent: Option<&'a mut Node<'a>>,
}

impl<'a> Node<'a> {
    fn new(v: i32) -> Self {
        Self {
            val: v,
            parent: None,
        }
    }

    fn new_child_env(&'a mut self, val: i32) -> Self {
        Self {
            val,
            parent: Some(self),
        }
    }

    fn lookup(&'a self, val: i32) -> bool {
        if self.val == val {
            true
        } else {
            match &self.parent {
                Some(parent) => parent.lookup(val),
                None => false,
            }
        }
    }
}

// Eval implements the evaluation rules for our interpreter.
fn eval<'a>(env: &'a mut Node<'a>, val: i32) -> bool {
    env.lookup(val)
}


#[derive(Debug)]
pub struct Program(pub Vec<Statement>);

#[derive(Debug)]
pub enum Expr {
    Var(String),
    I64(i64),
    Add(Box<Expr>, Box<Expr>),
    Mult(Box<Expr>, Box<Expr>),
    Read,
}

#[derive(Debug)]
pub enum Statement {
    VarDecl(String, Expr),
    Print(Expr),
    Switch(Switch),
    Block(Vec<Statement>),
    Break(Option<String>),
    While(Expr, Box<Statement>, Option<String>),
    Assignment(String, Expr),
}

#[derive(Debug)]
pub enum Case {
    Case(i64, Option<Statement>),
    Default(Statement),
}

#[derive(Debug)]
pub struct Switch {
    pub arg: Box<Expr>,
    pub cases: Vec<Case>,
}

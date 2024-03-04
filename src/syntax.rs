pub type SurfProg<Ann> = Exp<Ann>;
pub type SurfFunDecl<Ann> = FunDecl<Exp<Ann>, Ann>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunDecl<E, Ann> {
    pub name: String,
    pub parameters: Vec<String>,
    pub body: E,
    pub ann: Ann,
}

/* Expressions */
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exp<Ann> {
    Num(i64, Ann),
    Bool(bool, Ann),
    Var(String, Ann),
    Prim(Prim, Vec<Box<Exp<Ann>>>, Ann),
    Let {
        bindings: Vec<(String, Exp<Ann>)>, // new binding declarations
        body: Box<Exp<Ann>>,               // the expression in which the new variables are bound
        ann: Ann,
    },
    If {
        cond: Box<Exp<Ann>>,
        thn: Box<Exp<Ann>>,
        els: Box<Exp<Ann>>,
        ann: Ann,
    },

    Semicolon {
        e1: Box<Exp<Ann>>,
        e2: Box<Exp<Ann>>,
        ann: Ann,
    },

    FunDefs {
        decls: Vec<FunDecl<Exp<Ann>, Ann>>,
        body: Box<Exp<Ann>>,
        ann: Ann,
    },
    Lambda {
        parameters: Vec<String>,
        body: Box<Exp<Ann>>,
        ann: Ann,
    },
    MakeClosure {
        arity: usize,
        label: String,
        env: Box<Exp<Ann>>,
        ann: Ann,
    },

    // A call that may or may not require a closure
    Call(Box<Exp<Ann>>, Vec<Exp<Ann>>, Ann),

    // A call to a dynamically determined closure
    ClosureCall(Box<Exp<Ann>>, Vec<Exp<Ann>>, Ann),
    // A direct call to a known function definition
    DirectCall(String, Vec<Exp<Ann>>, Ann),

    // A local tail call to a static function
    InternalTailCall(String, Vec<Exp<Ann>>, Ann),
    // A global function call to either a static function or the code
    // pointer of a closure
    ExternalCall {
        fun: VarOrLabel,
        args: Vec<Exp<Ann>>,
        is_tail: bool,
        ann: Ann,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Prim {
    // unary
    Add1,
    Sub1,
    Not,
    Length,
    Print,
    IsBool,
    IsNum,
    IsFun,
    IsArray,
    // Internal-only unary forms
    GetCode,
    GetEnv,
    CheckArityAndUntag(usize),

    // binary
    Add,
    Sub,
    Mul,
    And,
    Or,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Neq,
    ArrayGet, // first arg is array, second is index

    // trinary
    ArraySet, // first arg is array, second is index, third is new value

    // 0 or more arguments
    MakeArray,
}

/* Sequential Expressions */
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqProg<Ann> {
    pub funs: Vec<FunDecl<SeqExp<Ann>, Ann>>,
    pub main: SeqExp<Ann>,
    pub ann: Ann,
}

pub type SeqFunDecl<Ann> = FunDecl<SeqExp<Ann>, Ann>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImmExp {
    Num(i64),
    Bool(bool),
    Var(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VarOrLabel {
    Var(String),
    Label(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeqExp<Ann> {
    Imm(ImmExp, Ann),
    Prim(Prim, Vec<ImmExp>, Ann),
    MakeClosure {
        arity: usize,
        label: String,
        env: ImmExp,
        ann: Ann,
    },
    Let {
        var: String,
        bound_exp: Box<SeqExp<Ann>>,
        body: Box<SeqExp<Ann>>,
        ann: Ann,
    },
    If {
        cond: ImmExp,
        thn: Box<SeqExp<Ann>>,
        els: Box<SeqExp<Ann>>,
        ann: Ann,
    },
    // Local function definitions
    // These should only be called using InternalTailCall
    FunDefs {
        decls: Vec<FunDecl<SeqExp<Ann>, Ann>>,
        body: Box<SeqExp<Ann>>,
        ann: Ann,
    },

    // An internal tail call to a locally defined function.
    // Implemented by setting arguments and then jmp in Assembly
    InternalTailCall(String, Vec<ImmExp>, Ann),

    // A call to one of the top-level function definitions
    // Uses the Snake Calling Convention v0
    // marked to indicate whether it is a tail call or not
    ExternalCall {
        fun: VarOrLabel,
        args: Vec<ImmExp>,
        is_tail: bool,
        ann: Ann,
    },

    Semicolon {
        e1: Box<SeqExp<Ann>>,
        e2: Box<SeqExp<Ann>>,
        ann: Ann,
    },
}

/* Useful functions for Exps, SeqExps */
impl<Ann> Exp<Ann> {
    pub fn ann(&self) -> Ann
        where
            Ann: Clone,
    {
        match self {
            Exp::Num(_, a)
            | Exp::Bool(_, a)
            | Exp::Var(_, a)
            | Exp::Prim(_, _, a)
            | Exp::Let { ann: a, .. }
            | Exp::If { ann: a, .. }
            | Exp::Call(_, _, a)
            | Exp::FunDefs { ann: a, .. }
            | Exp::InternalTailCall(_, _, a)
            | Exp::ExternalCall { ann: a, .. }
            | Exp::Semicolon { ann: a, .. }
            | Exp::Lambda { ann: a, .. }
            | Exp::MakeClosure { ann: a, .. }
            | Exp::ClosureCall(_, _, a)
            | Exp::DirectCall(_, _, a)
            => a.clone(),
        }
    }

    pub fn ann_mut(&mut self) -> &mut Ann {
        match self {
            Exp::Num(_, a)
            | Exp::Bool(_, a)
            | Exp::Var(_, a)
            | Exp::Prim(_, _, a)
            | Exp::Let { ann: a, .. }
            | Exp::If { ann: a, .. }
            | Exp::Call(_, _, a)
            | Exp::FunDefs { ann: a, .. }
            | Exp::InternalTailCall(_, _, a)
            | Exp::ExternalCall { ann: a, .. }
            | Exp::Semicolon { ann: a, .. }
            | Exp::Lambda { ann: a, .. }
            | Exp::MakeClosure { ann: a, .. }
            | Exp::ClosureCall(_, _, a)
            | Exp::DirectCall(_, _, a) => a,
        }
    }

    pub fn map_ann<Ann2, F>(&self, f: &mut F) -> Exp<Ann2>
        where
            F: FnMut(&Ann) -> Ann2,
    {
        match self {
            Exp::Num(n, a) => Exp::Num(*n, f(a)),
            Exp::Bool(b, a) => Exp::Bool(*b, f(a)),
            Exp::Var(s, a) => Exp::Var(s.clone(), f(a)),
            Exp::Prim(op, es, a) => Exp::Prim(
                *op,
                es.iter().map(|e| Box::new(e.map_ann(f))).collect(),
                f(a),
            ),
            Exp::Let {
                bindings,
                body,
                ann,
            } => Exp::Let {
                bindings: bindings
                    .iter()
                    .map(|(x, e)| (x.clone(), e.map_ann(f)))
                    .collect(),
                body: Box::new(body.map_ann(f)),
                ann: f(ann),
            },
            Exp::If {
                cond,
                thn,
                els,
                ann,
            } => Exp::If {
                cond: Box::new(cond.map_ann(f)),
                thn: Box::new(thn.map_ann(f)),
                els: Box::new(els.map_ann(f)),
                ann: f(ann),
            },
            Exp::Call(fun, args, ann) => Exp::Call(
                Box::new(fun.map_ann(f)),
                args.iter().map(|e| e.map_ann(f)).collect(),
                f(ann),
            ),
            Exp::InternalTailCall(fun, args, ann) => Exp::InternalTailCall(
                fun.clone(),
                args.iter().map(|e| e.map_ann(f)).collect(),
                f(ann),
            ),
            Exp::ExternalCall {
                fun,
                args,
                is_tail,
                ann,
            } => Exp::ExternalCall {
                fun: fun.clone(),
                args: args.iter().map(|e| e.map_ann(f)).collect(),
                is_tail: *is_tail,
                ann: f(ann),
            },
            Exp::FunDefs { decls, body, ann } => Exp::FunDefs {
                decls: decls.iter().map(|d| d.map_ann(f)).collect(),
                body: Box::new(body.map_ann(f)),
                ann: f(ann),
            },
            Exp::Semicolon { e1, e2, ann } => Exp::Semicolon {
                e1: Box::new(e1.map_ann(f)),
                e2: Box::new(e2.map_ann(f)),
                ann: f(ann)
            },
            Exp::Lambda { parameters, body, ann } => Exp::Lambda {
                parameters: parameters.clone(),
                body: Box::new(body.map_ann(f)),
                ann: f(ann)
            },
            Exp::MakeClosure { arity, label, env, ann } => Exp::MakeClosure {
                arity: *arity,
                label: label.clone(),
                env: Box::new(env.map_ann(f)),
                ann: f(ann)
            },
            Exp::ClosureCall(call, args, ann) => Exp::ClosureCall(
                Box::new(call.map_ann(f)),
                args.iter().map(|e| e.map_ann(f)).collect(),
                f(ann)
            ),
            Exp::DirectCall(call, args, ann) => Exp::DirectCall(
                call.clone(),
                args.iter().map(|e| e.map_ann(f)).collect(),
                f(ann)
            )
        }
    }
}

impl<Ann> SeqExp<Ann> {
    pub fn ann(&self) -> Ann
        where
            Ann: Clone,
    {
        match self {
            SeqExp::Imm(_, a)
            | SeqExp::Prim(_, _, a)
            | SeqExp::Let { ann: a, .. }
            | SeqExp::If { ann: a, .. }
            | SeqExp::InternalTailCall(_, _, a)
            | SeqExp::ExternalCall { ann: a, .. }
            | SeqExp::FunDefs { ann: a, .. }
            | SeqExp::Semicolon { ann: a, .. }
            | SeqExp::MakeClosure { ann: a, .. } => a.clone(),
        }
    }

    pub fn map_ann<Ann2, F>(&self, f: &mut F) -> SeqExp<Ann2>
        where
            F: FnMut(&Ann) -> Ann2,
    {
        match self {
            SeqExp::Imm(imm, a) => SeqExp::Imm(imm.clone(), f(a)),
            SeqExp::Prim(op, imms, a) => SeqExp::Prim(*op, imms.to_vec(), f(a)),
            SeqExp::Let {
                var,
                bound_exp,
                body,
                ann,
            } => SeqExp::Let {
                var: var.clone(),
                bound_exp: Box::new(bound_exp.map_ann(f)),
                body: Box::new(body.map_ann(f)),
                ann: f(ann),
            },
            SeqExp::If {
                cond,
                thn,
                els,
                ann,
            } => SeqExp::If {
                cond: cond.clone(),
                thn: Box::new(thn.map_ann(f)),
                els: Box::new(els.map_ann(f)),
                ann: f(ann),
            },
            SeqExp::InternalTailCall(fun, args, ann) => {
                SeqExp::InternalTailCall(fun.clone(), args.clone(), f(ann))
            }
            SeqExp::ExternalCall {
                fun,
                args,
                is_tail,
                ann,
            } => SeqExp::ExternalCall {
                fun: fun.clone(),
                args: args.clone(),
                is_tail: *is_tail,
                ann: f(ann),
            },
            SeqExp::FunDefs { decls, body, ann } => SeqExp::FunDefs {
                decls: decls.iter().map(|d| d.map_ann(f)).collect(),
                body: Box::new(body.map_ann(f)),
                ann: f(ann),
            },
            SeqExp::Semicolon { e1, e2, ann } => SeqExp::Semicolon {
                e1: Box::new(e1.map_ann(f)),
                e2: Box::new(e2.map_ann(f)),
                ann: f(ann)
            },
            SeqExp::MakeClosure { arity, label, env, ann } => SeqExp::MakeClosure {
                arity: *arity,
                label: label.clone(),
                env: env.clone(),
                ann: f(ann)
            }
        }
    }
}

impl<Ann> FunDecl<Exp<Ann>, Ann> {
    pub fn map_ann<Ann2, F>(&self, f: &mut F) -> FunDecl<Exp<Ann2>, Ann2>
        where
            F: FnMut(&Ann) -> Ann2,
    {
        FunDecl {
            name: self.name.clone(),
            parameters: self.parameters.clone(),
            body: self.body.map_ann(f),
            ann: f(&self.ann),
        }
    }
}

impl<Ann> SeqProg<Ann> {
    pub fn map_ann<Ann2, F>(&self, f: &mut F) -> SeqProg<Ann2>
        where
            F: FnMut(&Ann) -> Ann2,
    {
        SeqProg {
            funs: self.funs.iter().map(|d| d.map_ann(f)).collect(),
            main: self.main.map_ann(f),
            ann: f(&self.ann),
        }
    }
}
impl<Ann> FunDecl<SeqExp<Ann>, Ann> {
    pub fn map_ann<Ann2, F>(&self, f: &mut F) -> FunDecl<SeqExp<Ann2>, Ann2>
        where
            F: FnMut(&Ann) -> Ann2,
    {
        FunDecl {
            name: self.name.clone(),
            parameters: self.parameters.clone(),
            body: self.body.map_ann(f),
            ann: f(&self.ann),
        }
    }
}


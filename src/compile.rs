use crate::asm::{instrs_to_string, JmpArg, Offset};
use crate::asm::{Arg32, Arg64, BinArgs, Instr, MemRef, MovArgs, Reg, Reg32};
use crate::syntax::{Exp, FunDecl, ImmExp, Prim, SeqExp, SeqProg, SurfFunDecl, SurfProg, VarOrLabel};

use crate::list::List;

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use crate::syntax::Prim::MakeArray;

#[derive(Debug, PartialEq, Eq)]
pub enum CompileErr<Span> {
    UnboundVariable {
        unbound: String,
        location: Span,
    },
    UndefinedFunction {
        undefined: String,
        location: Span,
    },
    // The Span here is the Span of the let-expression that has the two duplicated bindings
    DuplicateBinding {
        duplicated_name: String,
        location: Span,
    },
    Overflow {
        num: i64,
        location: Span,
    },
    DuplicateFunName {
        duplicated_name: String,
        location: Span, // the location of the 2nd function
    },
    DuplicateArgName {
        duplicated_name: String,
        location: Span,
    },
}

static MAX_INT: i64 = i64::MAX >> 1;
static MIN_INT: i64 = i64::MIN >> 1;
static SCRATCH: Reg = Reg::R8;
static SCRATCH_ARRAY: Reg = Reg::R9;
static SCRATCH_INDEX: Reg = Reg::R10;
static SCRATCH_VALUE: Reg = Reg::R11;
static mut VAR: usize = 0;
static mut FUN: usize = 0;

fn tag_fun() -> usize {
    unsafe {
        let fun_tag = FUN;
        FUN += 1;
        fun_tag
    }
}

fn tag_var() -> usize {
    unsafe {
        let var_tag = VAR;
        VAR += 1;
        var_tag
    }
}

#[derive(Debug, Clone)]
struct SurfEnv<'e, FVal, VVal> {
    funs: Vec<(&'e str, FVal)>,
    vars: Vec<(&'e str, VVal)>,
}

pub fn get<T>(env: &[(&str, T)], x: &str) -> Option<T>
    where
        T: Clone,
{
    for (y, n) in env.iter().rev() {
        if x == *y {
            return Some(n.clone());
        }
    }
    None
}

impl<'e, FVal, VVal> SurfEnv<'e, FVal, VVal> {
    fn new() -> Self {
        SurfEnv {
            funs: Vec::new(),
            vars: Vec::new(),
        }
    }
    fn get_fun(&self, f: &'e str) -> Option<FVal>
        where
            FVal: Clone,
    {
        get(&self.funs, f)
    }
    fn push_fun(&mut self, x: &'e str, v: FVal) {
        self.funs.push((x, v))
    }

    fn get_var(&self, x: &'e str) -> Option<VVal>
        where
            VVal: Clone,
    {
        get(&self.vars, x)
    }
    fn push_var(&mut self, x: &'e str, v: VVal) {
        self.vars.push((x, v))
    }

    fn all_vars(&self) -> Vec<&'e str> {
        self.vars.iter().map(|(x, _)| *x).collect()
    }

    fn all_funs(&self) -> Vec<&'e str> {
        self.funs.iter().map(|(x, _)| *x).collect()
    }
}

fn check_exp<'exp, Span>(
    e: &'exp Exp<Span>,
    mut env: SurfEnv<'exp, usize, ()>,
) -> Result<(), CompileErr<Span>>
    where
        Span: Clone,
{
    match e {
        Exp::Num(n, span) => {
            // If a numeric constant is too large, report an Overflow error.
            if *n > MAX_INT || *n < MIN_INT {
                return Err(CompileErr::Overflow {
                    num: *n,
                    location: span.clone(),
                });
            }
        }

        Exp::Bool(_, _) => {}

        Exp::Var(x, span) => {
            if env.get_var(x).is_none() && env.get_fun(x).is_none() {
                return Err(CompileErr::UnboundVariable {
                    unbound: x.clone(),
                    location: span.clone(),
                })
            }
        }

        Exp::Prim(_, args, _) => {
            for arg in args {
                check_exp(arg, env.clone())?;
            }
        }

        Exp::Semicolon { e1, e2, .. } => {
            check_exp(e1, env.clone())?;
            check_exp(e2, env.clone())?;
        }

        Exp::Let { bindings, body, ann: span } => {
            let mut bound: HashSet<&str> = HashSet::new();
            for (x, e) in bindings {
                if bound.contains(x.as_str()) {
                    return Err(CompileErr::DuplicateBinding {
                        duplicated_name: String::from(x),
                        location: span.clone(),
                    });
                }

                check_exp(e, env.clone())?;
                bound.insert(x);
                env.push_var(x, ());
            }

            check_exp(body, env.clone())?;
        }

        Exp::If { cond, thn, els, .. } => {
            check_exp(cond, env.clone())?;
            check_exp(thn, env.clone())?;
            check_exp(els, env.clone())?;
        }

        Exp::FunDefs { decls, body, .. } => {
            let mut names: HashMap<&str, usize> = HashMap::new();
            for decl in decls {
                if names.contains_key(decl.name.as_str()) {
                    return Err(CompileErr::DuplicateFunName { 
                        duplicated_name: String::from(&decl.name), 
                        location: decl.ann.clone()
                    });
                }
                names.insert(decl.name.as_str(), decl.parameters.len());
            }

            for (name, size) in names.into_iter() {
                env.push_fun(name, size);
            }

            for decl in decls {
                let mut names: HashSet<&str> = HashSet::new();
                for p_name in decl.parameters.iter() {
                    if names.contains(p_name.as_str()) {
                        return Err(CompileErr::DuplicateArgName {
                            duplicated_name: p_name.to_string(),
                            location: decl.ann.clone(),
                        });
                    }
                    names.insert(p_name);
                    env.push_var(p_name, ());
                }
                check_exp(&decl.body, env.clone())?;
            }

            check_exp(body, env.clone())?;
        }
        
        Exp::Call(f, args, _) => {
            check_exp(f, env.clone())?;
            for arg in args {
                check_exp(arg, env.clone())?;
            }
        }

        Exp::Lambda { parameters, body, ann } => {
            let mut names: HashSet<&str> = HashSet::new();
            for p_name in parameters.iter() {
                if names.contains(p_name.as_str()) {
                    return Err(CompileErr::DuplicateArgName {
                        duplicated_name: p_name.to_string(),
                        location: ann.clone(),
                    });
                }
                names.insert(p_name);
                env.push_var(p_name, ());
            }
            check_exp(body, env)?;
        }

        Exp::MakeClosure { .. } | Exp::ClosureCall(..) | Exp::DirectCall(..) => {
            unreachable!("MakeClosure, ClosureCall, DirectCall should not happen in check prog...")
        }

        Exp::InternalTailCall(..) | Exp::ExternalCall { .. } => {
            unreachable!("ecall/icall should not happen in check prog...")
        }
    }
    Ok(())
}

pub fn check_prog<Span>(p: &SurfProg<Span>) -> Result<(), CompileErr<Span>>
    where
        Span: Clone,
{
    check_exp(p, SurfEnv::new())
}

fn uniquify(e: &Exp<u32>) -> Exp<()> {
    fn uniq_loop<'exp>(e: &'exp Exp<u32>, mut env: SurfEnv<'exp, String, String>) -> Exp<()> {
        match e {
            Exp::Num(n, _) => Exp::Num(*n, ()),
            Exp::Bool(b, _) => Exp::Bool(*b, ()),
            Exp::Prim(op, args, _) => Exp::Prim(
                *op,
                args.iter().map(|arg| Box::new(uniq_loop(arg, env.clone()))).collect(),
                ()
            ),
            Exp::If { cond, thn, els, .. } => Exp::If {
                cond: Box::new(uniq_loop(cond, env.clone())),
                thn: Box::new(uniq_loop(thn, env.clone())),
                els: Box::new(uniq_loop(els, env.clone())),
                ann: ()
            },
            Exp::Semicolon { e1, e2, .. } => Exp::Semicolon { 
                e1: Box::new(uniq_loop(e1, env.clone())), 
                e2: Box::new(uniq_loop(e2, env.clone())), 
                ann: () 
            },

            // Interesting cases: using a value/function variable
            Exp::Var(v, _) => {
                Exp::Var({
                    if let Some(var) = env.get_var(v) {
                        var
                    } else if let Some(fun) = env.get_fun(v) {
                        fun
                    } else {
                        unreachable!("Fun/var should exist in env by now...")
                    }
                }, ())
            }
            Exp::Call(f, args, _) => {
                // Eliminate Closures
                // DirectCall -> function names that are resolved to a statically known label
                // ClosureCall -> function being called is dynamically determined closure
                let new_args = args.iter().map(|arg| uniq_loop(arg, env.clone())).collect();
                match &**f {
                    Exp::Var(label, _) => {
                        if let Some(var) = env.get_fun(&label) {
                            return Exp::DirectCall(var, new_args, ());
                        }
                    }
                    _ => {}
                }
                Exp::ClosureCall(Box::new(uniq_loop(f, env.clone())), new_args, ())
            }

            // Declaring value / function variables
            Exp::Let { bindings, body, ann: noise } => {
                let new_bindings = bindings.iter().map(|(x, e)| {
                    let new_x = format!("{}#{}", x, noise);
                    let new_e = uniq_loop(e, env.clone());
                    env.push_var(x, new_x.clone());
                    (new_x, new_e)
                }).collect();
                
                Exp::Let {
                    bindings: new_bindings,
                    body: Box::new(uniq_loop(body, env)),
                    ann: (),
                }
            }

            Exp::FunDefs { decls, body, ann: noise } => {
                for decl in decls.iter() {
                    let new_f = format!("{}#{}", decl.name, noise);
                    env.push_fun(&decl.name, new_f);
                }

                let new_decls = decls.iter().map(|decl| {
                    let new_f = format!("{}#{}", decl.name, noise);
                    let mut local_env = env.clone();
                    let new_parameters = decl.parameters.iter().map(|parameter| {
                        let new_parameter = format!("{}#{}", parameter, noise);
                        local_env.push_var(parameter, new_parameter.clone());
                        new_parameter
                    }).collect();

                    FunDecl {
                        name: new_f,
                        parameters: new_parameters,
                        body: uniq_loop(&decl.body, local_env),
                        ann: (),
                    }
                }).collect();

                Exp::FunDefs { 
                    decls: new_decls, 
                    body: Box::new(uniq_loop(body, env)), 
                    ann: () 
                }
            }

            Exp::Lambda { parameters, body, ann } => {
                let mut local_env = env.clone();
                let new_parameters = parameters.iter().map(
                    |param| {
                        let new_param = format!("{}#{}", param, ann);
                        local_env.push_var(param, new_param.clone());
                        new_param
                    }
                ).collect();

                Exp::Lambda {
                    parameters: new_parameters,
                    body: Box::new(uniq_loop(body, local_env)),
                    ann: ()
                }
            }

            Exp::MakeClosure { .. } | Exp::ClosureCall(..) | Exp::DirectCall(..) => {
                unreachable!("MakeClosure, ClosureCall, DirectCall should not happen in uniquify...")
            }
            
            // Impossible cases: should never happen in this pass
            Exp::InternalTailCall(..) | Exp::ExternalCall{..} => {
                unreachable!("ecall/icall should never happen in uniquify...")
            }
        }
    }
    uniq_loop(e, SurfEnv::new())
}

// Analyzes an expression to determine which functions within it need
// to be lambda lifted.

// What should be lambda lifted?
// 1. Any function that is called with a non-tail call.
// 2. Any function that is live in another function that is lambda lifted.
// Oversimplification of 2: any function that is in scope in another function that is lambda lifted.
fn should_lifts<Ann>(p: &Exp<Ann>) -> HashSet<String> {
    fn sl<'e, Ann>(
        p: &'e Exp<Ann>,
        mut env: SurfEnv<'e, Vec<&'e str>, ()>,
        shoulds: &mut HashSet<String>,
        is_tail: bool,
    ) {
        // is_tail: is the expression in a tail position?
        // lift any LOCAL function definition that is within the scope of another being lifted...

        match p {
            Exp::Num(..) | Exp::Bool(..) | Exp::Var(..) => {}

            Exp::Prim(_, args, _) => {
                for arg in args {
                    sl(arg, env.clone(), shoulds, false)
                }
            }

            Exp::Let {
                bindings,
                body,
                ..
            } => {
                for (_, rhs) in bindings {
                    sl(rhs, env.clone(), shoulds, false);
                }
                sl(body, env, shoulds, is_tail);
            }

            Exp::If {
                cond,
                thn,
                els,
                ..
            } => {
                sl(cond, env.clone(), shoulds, false);
                sl(thn, env.clone(), shoulds, is_tail);
                sl(els, env.clone(), shoulds, is_tail);
            }

            Exp::Semicolon {
                e1,
                e2,
                ..
            } => {
                sl(e1, env.clone(), shoulds, false);
                sl(e2, env.clone(), shoulds, is_tail);
            }

            Exp::FunDefs {
                decls,
                body,
                ..
            } => {
                let mut live_funs = env.all_funs();
                live_funs.extend(decls.iter().map(|d| d.name.as_str()));
                for decl in decls {
                    env.push_fun(&decl.name, live_funs.clone());
                }
                for decl in decls {
                    sl(&decl.body, env.clone(), shoulds, true);
                }
                sl(body, env.clone(), shoulds, is_tail);
            }

            Exp::Lambda {
                parameters,
                body,
                ..
            } => {
                for param in parameters {
                    env.push_var(param, ());
                }

                sl(body, env, shoulds, true);
            }

            Exp::ClosureCall(f, args, ..) => {
                match &**f {
                    Exp::Var(fun_name, _) => {
                        shoulds.insert(fun_name.clone());
                    }
                    _ => {}
                }

                sl(f, env.clone(), shoulds, true);

                for arg in args {
                    sl(arg, env.clone(), shoulds, false);
                }
            }

            Exp::DirectCall(fun_name, args, ..) => {
                for arg in args {
                    sl(arg, env.clone(), shoulds, false);
                }

                if !is_tail {
                    for g in env.get_fun(fun_name).unwrap_or_else(
                        || panic!("Function name expected in env for direct calls...")
                    ) {
                        shoulds.insert(g.to_string());
                    }
                }
            }

            Exp::MakeClosure { .. } | Exp::Call(..) => {
                unreachable!("MakeClosure, Call should not happen in uniquify...")
            }

            Exp::InternalTailCall(..) | Exp::ExternalCall { .. } => {
                unreachable!("ecall/icall should never happen in lambda lifting...")
            }
        }
    }
    let mut hs = HashSet::new();
    sl(p, SurfEnv::new(), &mut hs, true);
    hs
}

// Precondition: all name bindings are unique
fn lambda_lift<Ann>(p: &Exp<Ann>) -> (Vec<FunDecl<Exp<()>, ()>>, Exp<()>) {
    fn ll<'exp, Ann>(
        should_be_lifted: &HashSet<String>,
        e: &'exp Exp<Ann>,
        o_decls: &mut Vec<FunDecl<Exp<()>, ()>>,
        mut env: SurfEnv<'exp, Vec<String>, String>,
        tail_position: bool,
    ) -> Exp<()> {
        match e {
            Exp::Num(n, _) => {
                Exp::Num(*n, ())
            }

            Exp::Bool(b, _) => {
                Exp::Bool(*b, ())
            }

            Exp::Var(v, _) => {
                Exp::Var(v.clone(), ())
            }

            Exp::Prim(op, args, _) => {
                Exp::Prim(
                    *op,
                    args.iter().map(
                        |e| Box::new(ll(should_be_lifted, e, o_decls, env.clone(), false))
                    ).collect(),
                    ()
                )
            }

            Exp::Let {
                bindings,
                body,
                ..
            } => {
                let new_bindings = bindings.iter().map(
                    |(x, rhs)| {
                        let e = ll(should_be_lifted, rhs, o_decls, env.clone(), false);
                        match e.clone() {
                            Exp::MakeClosure { label, .. } => {
                                eprintln!("PETER: Mapped {} -> {}", x.clone(), label.clone());
                                env.push_var(x, label.clone());
                            }
                            _ => {
                                env.push_var(x, String::from(""));
                            }
                        }
                        (x.clone(), e)
                    }
                ).collect();

                Exp::Let {
                    bindings: new_bindings,
                    body: Box::new(ll(should_be_lifted, body, o_decls, env.clone(), tail_position)),
                    ann: ()
                }
            }

            Exp::If {
                cond,
                thn,
                els,
                ..
            } => {
                Exp::If {
                    cond: Box::new(ll(should_be_lifted, cond, o_decls, env.clone(), false)),
                    thn: Box::new(ll(should_be_lifted, thn, o_decls, env.clone(), tail_position)),
                    els: Box::new(ll(should_be_lifted, els, o_decls, env.clone(), tail_position)),
                    ann: ()
                }
            }

            Exp::Semicolon {
                e1,
                e2,
                ..
            } => {
                Exp::Semicolon {
                    e1: Box::new(ll(should_be_lifted, e1, o_decls, env.clone(), false)),
                    e2: Box::new(ll(should_be_lifted, e2, o_decls, env.clone(), tail_position)),
                    ann: ()
                }
            }

            Exp::FunDefs {
                decls,
                body,
                ..
            } => {
                // Most interesting case :)
                // First, we figure out what the current variables are...
                let free_vars: Vec<String> = env.all_vars().into_iter().map(String::from).collect();

                // Extend the environment for function -> all vars
                for decl in decls {
                    if should_be_lifted.contains(&decl.name) {
                        env.push_fun(&decl.name, free_vars.clone());
                    }
                }

                let mut unlifted_decls: Vec<FunDecl<Exp<()>, ()>> = Vec::new();
                for decl in decls {
                    if should_be_lifted.contains(&decl.name) {
                        // If the function is to be lifted, then
                        // 1. Combine free variables with parameters [other vars, parameters]
                        // 2. Update the local env to include (1)
                        // 3. Lambda lift the body
                        // 4. Add to global function definition
                        let mut new_params = free_vars.clone();
                        new_params.extend(decl.parameters.iter().map(String::from));
                        let mut local_env = env.clone();
                        for param in new_params.iter() {
                            local_env.push_var(param, String::from(""));
                        }
                        let fd = FunDecl {
                            name: decl.name.clone(),
                            parameters: new_params.clone(),
                            body: ll(should_be_lifted, &decl.body, o_decls, local_env, true),
                            ann: ()
                        };
                        o_decls.push(fd);
                        // TODO: Add env to unlifted_decls, f_closure to o_decls
                    } else {
                        // If the function is a tail call (i.e. does not need to be lifted), then
                        // simply lambda lift its body and add to the main function declaration
                        let fd = FunDecl {
                            name: decl.name.clone(),
                            parameters: decl.parameters.clone(),
                            body: ll(should_be_lifted, &decl.body, o_decls, env.clone(), true),
                            ann: ()
                        };
                        unlifted_decls.push(fd);
                    }
                }

                // Return the updated body
                Exp::FunDefs {
                    decls: unlifted_decls,
                    body: Box::new(ll(should_be_lifted, body, o_decls, env.clone(), tail_position)),
                    ann: (),
                }
            }

            Exp::Lambda {
                parameters,
                body,
                ..
            } => {
                let tag = tag_fun();
                let lambda_name = format!("closure_call#{}", tag);
                let env_name = format!("env#{}", tag);
                let free_vars: Vec<String> = env.all_vars().into_iter().map(String::from).collect();

                let mut local_env = env.clone();
                for param in parameters {
                    local_env.push_var(param, String::from(""));
                }

                let mut lifted_param = vec![env_name.clone()];
                lifted_param.extend(parameters.clone());
                let mut new_bindings = vec![];
                for (i, free_var) in free_vars.iter().enumerate() {
                    new_bindings.push(
                        (
                            free_var.clone(),
                            Exp::Prim(
                                Prim::ArrayGet,
                                vec![
                                    Box::new(Exp::Var(env_name.clone(), ())),
                                    Box::new(Exp::Num(i as i64, ()))
                                ],
                                ()
                            )
                        )
                    );
                }
                let lifted_lambda = FunDecl{
                    name: lambda_name.clone(),
                    parameters: lifted_param.clone(),
                    body: Exp::Let {
                        bindings: new_bindings,
                        body: Box::new(ll(should_be_lifted, body, o_decls, local_env, true)),
                        ann: ()
                    },
                    ann: ()
                };
                o_decls.push(lifted_lambda);

                Exp::MakeClosure {
                    arity: parameters.len(),
                    label: lambda_name.clone(),
                    env: Box::new(Exp::Prim(
                        MakeArray,
                        free_vars.into_iter().map(|free_var| {
                            Box::new(Exp::Var(free_var.clone(), ()))
                        }).collect(),
                        ()
                    )),
                    ann: ()
                }
            }

            Exp::ClosureCall(closure, args, _) => {
                let lifted_closure = ll(should_be_lifted, closure, o_decls, env.clone(), true);
                if let Exp::Var(name, ..) = lifted_closure {
                    let closure_call = env.get_var(name.as_str()).unwrap();
                    eprintln!("PETER: Transformed {} to: {}", name, closure_call);
                    Exp::Let {
                        bindings: vec![
                            (
                                closure_call.clone(),
                                Exp::Var(name, ())
                            )
                        ],
                        body: Box::new(Exp::ClosureCall(
                            Box::new(Exp::Var(closure_call, ())),
                            args.iter().map(|arg| {
                                ll(should_be_lifted, arg, o_decls, env.clone(), false)
                            }).collect(),
                            ()
                        )),
                        ann: ()
                    }
                } else {
                    panic!("DIDN'T FIND THE CLOSURE MAPPING!!!");
                }
            }

            Exp::DirectCall(fun_name, args, ..) => {
                if should_be_lifted.contains(fun_name) {
                    let mut new_args: Vec<Exp<()>> = env
                        .get_fun(fun_name)
                        .unwrap()
                        .into_iter()
                        .map(|x| Exp::Var(x.clone(), ()))
                        .collect();

                    new_args.extend(
                        args.iter().map(
                            |e| ll(should_be_lifted, e, o_decls, env.clone(), false)
                        )
                    );

                    Exp::ExternalCall {
                        fun: VarOrLabel::Label(fun_name.to_string()),
                        args: new_args,
                        is_tail: tail_position,
                        ann: ()
                    }
                } else if !tail_position {
                    panic!("Non-tail call to a function that hasn't been lambda lifted...")
                } else {
                    Exp::InternalTailCall(
                        fun_name.to_string(),
                        args.iter().map(
                            |e| ll(should_be_lifted, e, o_decls, env.clone(), false)
                        ).collect(),
                        ()
                    )
                }
            }

            Exp::MakeClosure { .. } => {
                unreachable!("MakeClosure shouldn't exist in sequentialize yet...")
            }

            Exp::Call(..) => {
                unreachable!("Call shouldn't reach sequentialize...")
            }

            Exp::InternalTailCall(..) | Exp::ExternalCall { .. } => {
                unreachable!("ecall/icall should not happen in lambda lifting...")
            }
        }
    }
    let should_be_lifted = should_lifts(p);

    let mut v = Vec::new();
    let e = ll(&should_be_lifted, p, &mut v, SurfEnv::new(), true);
    (v, e)
}

fn tag_exp<Ann>(p: &SurfProg<Ann>) -> SurfProg<u32> {
    let mut i = 0;
    p.map_ann(
        &mut (|_| {
            let cur = i;
            i += 1;
            cur
        }),
    )
}

fn tag_prog<Ann>(
    defs: &[FunDecl<Exp<Ann>, Ann>],
    main: &Exp<Ann>,
) -> (Vec<FunDecl<Exp<u32>, u32>>, Exp<u32>) {
    let mut i = 0;
    (
        defs.iter()
            .map(|decl| {
                decl.map_ann(
                    &mut (|_| {
                        let cur = i;
                        i += 1;
                        cur
                    }),
                )
            })
            .collect(),
        main.map_ann(
            &mut (|_| {
                let cur = i;
                i += 1;
                cur
            }),
        ),
    )
}

fn tag_sprog<Ann>(p: &SeqProg<Ann>) -> SeqProg<u32> {
    let mut i = 0;
    p.map_ann(
        &mut (|_| {
            let cur = i;
            i += 1;
            cur
        }),
    )
}

fn sequentialize(e: &Exp<u32>) -> SeqExp<()> {
    match e {
        Exp::Num(i, _) => SeqExp::Imm(ImmExp::Num(*i), ()),
        Exp::Bool(b, _) => SeqExp::Imm(ImmExp::Bool(*b), ()),
        Exp::Var(x, _) => SeqExp::Imm(ImmExp::Var(x.clone()), ()),

        Exp::Prim(op, es, ann) => {
            let vars: Vec<String> = es
                .iter()
                .enumerate()
                .map(|(sz, _)| format!("prim_var#{}#{}", sz, ann))
                .collect();
            let mut body = SeqExp::Prim(
                *op,
                vars.iter().map(|x| ImmExp::Var(x.clone())).collect(),
                (),
            );
            for (var, e) in vars.into_iter().zip(es).rev() {
                body = SeqExp::Let {
                    var,
                    bound_exp: Box::new(sequentialize(e)),
                    body: Box::new(body),
                    ann: (),
                }
            }
            body
        }

        Exp::Let { bindings, body, .. } => {
            let mut s_e = sequentialize(body);
            for (x, e) in bindings.iter().rev() {
                s_e = SeqExp::Let {
                    var: x.clone(),
                    bound_exp: Box::new(sequentialize(e)),
                    body: Box::new(s_e),
                    ann: (),
                }
            }
            s_e
        }

        Exp::If {
            cond,
            thn,
            els,
            ann: tag,
        } => {
            let s_cond = sequentialize(cond);
            let name = format!("#if_{}", tag);
            SeqExp::Let {
                var: name.clone(),
                bound_exp: Box::new(s_cond),
                body: Box::new(SeqExp::If {
                    cond: ImmExp::Var(name),
                    thn: Box::new(sequentialize(thn)),
                    els: Box::new(sequentialize(els)),
                    ann: (),
                }),
                ann: (),
            }
        }

        Exp::InternalTailCall(name, args, tag) => {
            let arg_vars: Vec<String> = (0..args.len())
                .map(|i| format!("#internal_tail_call_arg_{}_{}", tag, i))
                .collect();
            let mut body = SeqExp::InternalTailCall(
                name.clone(),
                arg_vars.iter().map(|x| ImmExp::Var(x.clone())).collect(),
                (),
            );
            for (arg_e, arg_var) in args.iter().zip(arg_vars).rev() {
                body = SeqExp::Let {
                    var: arg_var,
                    bound_exp: Box::new(sequentialize(arg_e)),
                    body: Box::new(body),
                    ann: (),
                };
            }
            body
        }

        Exp::ExternalCall {
            fun,
            args,
            is_tail,
            ann: tag,
        } => {
            let arg_vars: Vec<String> = (0..args.len())
                .map(|i| format!("#external_call_arg_{}_{}", tag, i))
                .collect();
            let mut body = SeqExp::ExternalCall {
                fun: fun.clone(),
                args: arg_vars.iter().map(|x| ImmExp::Var(x.clone())).collect(),
                is_tail: *is_tail,
                ann: (),
            };
            for (arg_e, arg_var) in args.iter().zip(arg_vars).rev() {
                body = SeqExp::Let {
                    var: arg_var,
                    bound_exp: Box::new(sequentialize(arg_e)),
                    body: Box::new(body),
                    ann: (),
                };
            }
            body
        }

        Exp::FunDefs { decls, body, .. } => SeqExp::FunDefs {
            decls: decls
                .iter()
                .map(|d| FunDecl {
                    name: d.name.clone(),
                    parameters: d.parameters.clone(),
                    body: sequentialize(&d.body),
                    ann: (),
                })
                .collect(),
            body: Box::new(sequentialize(body)),
            ann: (),
        },

        Exp::Semicolon {
            e1,
            e2,
            ann
        } => {
            // e1 ; e2 -> let DONT_CARE = e1 in e2
            SeqExp::Let {
                var: format!("DONT_CARE#{}", ann),
                bound_exp: Box::new(sequentialize(e1)),
                body: Box::new(sequentialize(e2)),
                ann: ()
            }
        }

        Exp::MakeClosure {
            arity,
            label,
            env,
            ann: tag
        } => {
            let env_name = format!("env#{}", tag);
            SeqExp::Let {
                var: env_name.clone(),
                bound_exp: Box::new(sequentialize(env)),
                body: Box::new(SeqExp::MakeClosure {
                    arity: *arity,
                    label: label.clone(),
                    env: ImmExp::Var(env_name.clone()),
                    ann: ()
                }),
                ann: ()
            }
        }

        Exp::ClosureCall(closure, args, tag) => {
            let fun_name = {
                match sequentialize(closure) {
                    SeqExp::Imm(imm, ..) => {
                        match imm {
                            ImmExp::Var(closure_name) => {
                                closure_name
                            }
                            _ => panic!("Closure not in variable form...")
                        }
                    }

                    _ => panic!("Closure not in variable form...")
                }
            };
            let untagged = format!("untagged#{}", tag.clone());
            let env_name = format!("env#{}", tag.clone());
            let mut new_args = vec![Exp::Var(env_name.clone(), tag.clone())];
            new_args.extend(args.iter().map(|arg| arg.clone()));
            sequentialize(&Exp::Let {
                bindings: vec![
                    (
                        untagged.clone(),
                        Exp::Prim(
                            Prim::CheckArityAndUntag(args.len()),
                            vec![Box::new(Exp::Var(fun_name.clone(), *tag))],
                            tag.clone()
                        )
                    ),
                    (
                        fun_name.clone(),
                        Exp::Prim(
                            Prim::GetCode,
                            vec![Box::new(Exp::Var(untagged.clone(), tag.clone()))],
                            tag.clone()
                        )
                    ),
                    (
                        env_name.clone(),
                        Exp::Prim(
                            Prim::GetEnv,
                            vec![Box::new(Exp::Var(untagged.clone(), tag.clone()))],
                            tag.clone()
                        )
                    )
                ],
                body: Box::new(Exp::ExternalCall {
                    fun: VarOrLabel::Var(fun_name),
                    args: new_args,
                    is_tail: false,
                    ann: tag.clone(),
                }),
                ann: tag.clone()
            })
        }

        Exp::DirectCall(label, args, ..) => {
            unreachable!("Direct call shouldn't reach sequentialize")
        }

        Exp::Call(..) => {
            unreachable!("Call shouldn't reach sequentialize")
        }

        Exp::Lambda { .. } => {
            unreachable!("Lambda shouldn't reach sequentialize")
        }
    }
}

fn seq_prog(decls: &[FunDecl<Exp<u32>, u32>], p: &Exp<u32>) -> SeqProg<()> {
    SeqProg {
        funs: decls
            .iter()
            .map(|d| FunDecl {
                name: d.name.clone(),
                parameters: d.parameters.clone(),
                body: sequentialize(&d.body),
                ann: (),
            })
            .collect(),
        main: sequentialize(p),
        ann: (),
    }
}

static SNAKE_TRUE: u64 = 0xFF_FF_FF_FF_FF_FF_FF_FF;
static SNAKE_FALSE: u64 = 0x7F_FF_FF_FF_FF_FF_FF_FF;
static NOT_MASK: u64 = 0x80_00_00_00_00_00_00_00;
static HEAP_PTR: Reg = Reg::R15; // next available heap location

fn compile_imm(e: &ImmExp, env: &CodeGenEnv) -> Arg64 {
    match e {
        ImmExp::Num(i) => Arg64::Signed(*i << 1),
        ImmExp::Bool(b) => Arg64::Unsigned(if *b { SNAKE_TRUE } else { SNAKE_FALSE }),
        ImmExp::Var(x) => Arg64::Mem(MemRef {
            reg: Reg::Rsp,
            offset: Offset::Constant(
                env.lookup_local_offset(x)
                    .unwrap_or_else(|| panic!("didn't find {}", x)),
            ),
        }),
    }
}

fn cmp_args<F>(args: BinArgs, cond_jmp: F, tag: u32) -> Vec<Instr>
    where
        F: FnOnce(JmpArg) -> Instr,
{
    let tru_lab = format!("cmp_true#{}", tag);
    let done_lab = format!("cmp_done#{}", tag);
    vec![
        Instr::Cmp(args),
        cond_jmp(JmpArg::Label(tru_lab.clone())),
        Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_FALSE))),
        Instr::Jmp(JmpArg::Label(done_lab.clone())),
        Instr::Label(tru_lab),
        Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_TRUE))),
        Instr::Label(done_lab),
    ]
}

fn num_test(arg: Arg64, lab: &str) -> Vec<Instr> {
    vec![
        Instr::Mov(MovArgs::ToReg(Reg::Rsi, arg)),
        Instr::Test(BinArgs::ToReg(Reg::Rsi, Arg32::Signed(1))),
        Instr::Jnz(JmpArg::Label(String::from(lab))),
    ]
}

fn bool_test(arg: Arg64, lab: &str) -> Vec<Instr> {
    vec![
        Instr::Mov(MovArgs::ToReg(Reg::Rsi, arg)),
        Instr::Test(BinArgs::ToReg(Reg::Rsi, Arg32::Signed(1))),
        Instr::Jz(JmpArg::Label(String::from(lab))),
    ]
}

fn array_test(e: &ImmExp, env: &CodeGenEnv) -> Vec<Instr> {
    vec![
        Instr::Mov(MovArgs::ToReg(Reg::Rax, compile_imm(e, env))),
        Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Reg(Reg::Rax))),
        Instr::And(BinArgs::ToReg(SCRATCH, Arg32::Unsigned(7))),
        Instr::Cmp(BinArgs::ToReg(SCRATCH, Arg32::Unsigned(1))),
    ]
}

fn closure_test(e: &ImmExp, env: &CodeGenEnv) -> Vec<Instr> {
    vec![
        Instr::Mov(MovArgs::ToReg(Reg::Rax, compile_imm(e, env))),
        Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Reg(Reg::Rax))),
        Instr::And(BinArgs::ToReg(SCRATCH, Arg32::Unsigned(7))),
        Instr::Cmp(BinArgs::ToReg(SCRATCH, Arg32::Unsigned(3))),
    ]
}

fn prim2_num_checks(arg1: Arg64, arg2: Arg64, lab: &str) -> Vec<Instr> {
    let mut is = num_test(arg1, lab);
    is.extend(num_test(arg2, lab));
    is
}
fn prim2_bool_checks(arg1: Arg64, arg2: Arg64, lab: &str) -> Vec<Instr> {
    let mut is = bool_test(arg1, lab);
    is.extend(bool_test(arg2, lab));
    is
}

fn user_fun_to_label(s: &str) -> String {
    format!("user_{}", s)
}

fn mov_to_mem64(mem: MemRef, arg: Arg64) -> Vec<Instr> {
    match arg {
        Arg64::Reg(r) => vec![Instr::Mov(MovArgs::ToMem(mem, Reg32::Reg(r)))],
        Arg64::Mem(mem_from) => vec![
            Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Mem(mem_from))),
            Instr::Mov(MovArgs::ToMem(mem, Reg32::Reg(SCRATCH))),
        ],
        Arg64::Signed(n) => {
            vec![
                Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Signed(n))),
                Instr::Mov(MovArgs::ToMem(mem, Reg32::Reg(SCRATCH))),
            ]
        }
        Arg64::Unsigned(n) => {
            vec![
                Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Unsigned(n))),
                Instr::Mov(MovArgs::ToMem(mem, Reg32::Reg(SCRATCH))),
            ]
        }
        _ => panic!("TODO: Mov to Mem64")
    }
}

#[derive(Clone, Debug)]
struct CodeGenEnv<'e> {
    bb_offsets: List<(&'e str, i32)>,
    local_offsets: List<(&'e str, i32)>,
    size: i32,
}

impl<'e> CodeGenEnv<'e> {
    fn new(params: Vec<(&'e str, i32)>) -> CodeGenEnv<'e> {
        CodeGenEnv {
            bb_offsets: List::new(),
            local_offsets: params.into_iter().collect(),
            size: 0,
            // next: -8,
        }
    }

    fn push_fun(&mut self, f: &'e str) {
        self.bb_offsets.push((f, self.size))
    }

    // if currently I have size variables, the next variable will be stored at offset rsp - 8 * size
    fn push(&mut self, x: &'e str) -> i32 {
        self.size += 1;
        let new = self.size * -8;
        self.local_offsets.push((x, new));
        new
    }

    fn lookup_local_offset(&self, x: &str) -> Option<i32> {
        for (y, off) in self.local_offsets.iter() {
            if x == y {
                return Some(off);
            }
        }
        None
    }

    fn lookup_fun_offset(&self, f: &str) -> Option<i32> {
        for (x, off) in self.bb_offsets.iter() {
            if x == f {
                return Some(off);
            }
        }
        None
    }
}

fn round_up_even(i: i32) -> i32 {
    i + (i % 2)
}

fn round_up_odd(i: i32) -> i32 {
    i + ((i + 1) % 2)
}

// Error Checking Cases
// (1) Need to have a dynamic error when a number, boolean, or array is used
//    as a function in a call. Error with message substr "called a non-function".
// (2) Delay FunctionCalledWrongArity to runtime - when a function is called with the wrong
//     number of arguments at runtime, error with message substr "wrong number of arguments".
fn compile_with_env<'exp>(
    e: &'exp SeqExp<u32>,      // the expression to be compiled
    mut env: CodeGenEnv<'exp>, // the environment mapping variables to their location on the stack and local functions to the offset to their arguments on the stack
) -> Vec<Instr> {
    match e {
        SeqExp::Imm(imm, _) => {
            vec![Instr::Mov(MovArgs::ToReg(Reg::Rax, compile_imm(imm, &env)))]
        }

        SeqExp::Prim(op, imms, tag) => match op {
            Prim::Add1 | Prim::Sub1 | Prim::Not | Prim::Print | Prim::IsBool | Prim::IsNum => {
                let imm = &imms[0];
                let arg = compile_imm(imm, &env);
                let mut is = vec![];
                match op {
                    Prim::Add1 => {
                        is.extend(num_test(arg.clone(), "arith_err"));
                        is.push(Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(Reg::Rsi))));
                        is.extend(vec![
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, arg.clone())),
                            Instr::Add(BinArgs::ToReg(Reg::Rax, Arg32::Signed(2))),
                            Instr::Jo(JmpArg::Label(String::from("overflow_err"))),
                        ])
                    }
                    Prim::Sub1 => {
                        is.extend(num_test(arg.clone(), "arith_err"));
                        is.push(Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(Reg::Rsi))));
                        is.extend(vec![
                            Instr::Sub(BinArgs::ToReg(Reg::Rax, Arg32::Signed(2))),
                            Instr::Jo(JmpArg::Label(String::from("overflow_err"))),
                        ])
                    }
                    Prim::Not => {
                        is.extend(bool_test(arg.clone(), "log_err"));
                        is.push(Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(Reg::Rsi))));
                        is.extend(vec![
                            Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Unsigned(NOT_MASK))),
                            Instr::Xor(BinArgs::ToReg(Reg::Rax, Arg32::Reg(SCRATCH))),
                        ]);
                    }
                    Prim::IsBool => {
                        let tru_lab = format!("isbool_true#{}", tag);
                        let done_lab = format!("isnum_done#{}", tag);
                        is.extend(vec![
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, arg)),
                            Instr::Test(BinArgs::ToReg(Reg::Rax, Arg32::Signed(1))),
                            Instr::Jnz(JmpArg::Label(tru_lab.clone())),
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_FALSE))),
                            Instr::Jmp(JmpArg::Label(done_lab.clone())),
                            Instr::Label(tru_lab),
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_TRUE))),
                            Instr::Label(done_lab),
                        ])
                    }
                    Prim::IsNum => {
                        let tru_lab = format!("isbool_true#{}", tag);
                        let done_lab = format!("isnum_done#{}", tag);
                        is.extend(vec![
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, arg)),
                            Instr::Test(BinArgs::ToReg(Reg::Rax, Arg32::Signed(1))),
                            Instr::Jz(JmpArg::Label(tru_lab.clone())),
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_FALSE))),
                            Instr::Jmp(JmpArg::Label(done_lab.clone())),
                            Instr::Label(tru_lab),
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_TRUE))),
                            Instr::Label(done_lab),
                        ])
                    }
                    Prim::Print => is.extend(vec![
                        Instr::Mov(MovArgs::ToReg(Reg::Rdi, arg)),
                        Instr::Sub(BinArgs::ToReg(
                            Reg::Rsp,
                            Arg32::Signed(8 * round_up_even(env.size)),
                        )),
                        Instr::Call(JmpArg::Label("print_snake_val".to_string())),
                        Instr::Add(BinArgs::ToReg(
                            Reg::Rsp,
                            Arg32::Signed(8 * round_up_even(env.size)),
                        )),
                    ]),
                    _ => unreachable!(),
                };
                is
            }
            Prim::Add
            | Prim::Sub
            | Prim::Mul
            | Prim::And
            | Prim::Or
            | Prim::Lt
            | Prim::Gt
            | Prim::Le
            | Prim::Ge
            | Prim::Eq
            | Prim::Neq => {
                let imm1 = &imms[0];
                let imm2 = &imms[1];
                let arg1 = compile_imm(imm1, &env);
                let arg2 = compile_imm(imm2, &env);
                let mut is = vec![
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, arg1.clone())),
                    Instr::Mov(MovArgs::ToReg(SCRATCH, arg2.clone())),
                ];
                let args = BinArgs::ToReg(Reg::Rax, Arg32::Reg(SCRATCH));
                match op {
                    Prim::Add => {
                        is.extend(prim2_num_checks(arg1, arg2, "arith_err"));
                        is.push(Instr::Add(args));
                        is.push(Instr::Jo(JmpArg::Label(String::from("overflow_err"))));
                    }
                    Prim::Sub => {
                        is.extend(prim2_num_checks(arg1, arg2, "arith_err"));
                        is.push(Instr::Sub(args));
                        is.push(Instr::Jo(JmpArg::Label(String::from("overflow_err"))));
                    }
                    Prim::Mul => {
                        is.extend(prim2_num_checks(arg1, arg2, "arith_err"));
                        is.extend(vec![
                            Instr::IMul(args),
                            Instr::Jo(JmpArg::Label(String::from("overflow_err"))),
                            Instr::Sar(BinArgs::ToReg(Reg::Rax, Arg32::Signed(1))),
                        ]);
                    }
                    Prim::And => {
                        is.extend(prim2_bool_checks(arg1, arg2, "log_err"));
                        is.extend(vec![Instr::And(args)])
                    }
                    Prim::Or => {
                        is.extend(prim2_bool_checks(arg1, arg2, "log_err"));
                        is.extend(vec![Instr::Or(args)])
                    }
                    Prim::Lt => {
                        is.extend(prim2_num_checks(arg1, arg2, "cmp_err"));
                        is.extend(cmp_args(args, Instr::Jl, *tag));
                    }
                    Prim::Le => {
                        is.extend(prim2_num_checks(arg1, arg2, "cmp_err"));
                        is.extend(cmp_args(args, Instr::Jle, *tag));
                    }
                    Prim::Gt => {
                        is.extend(prim2_num_checks(arg1, arg2, "cmp_err"));
                        is.extend(cmp_args(args, Instr::Jg, *tag));
                    }
                    Prim::Ge => {
                        is.extend(prim2_num_checks(arg1, arg2, "cmp_err"));
                        is.extend(cmp_args(args, Instr::Jge, *tag));
                    }
                    Prim::Neq => {
                        is.extend(cmp_args(args, Instr::Jne, *tag));
                    }
                    Prim::Eq => {
                        is.extend(cmp_args(args, Instr::Je, *tag));
                    }
                    _ => unreachable!(),
                }
                is
            }

            Prim::MakeArray => {
                let mut is = vec![];

                // Move the size of the array onto the heap at [R15 + 8 * 0]
                is.extend(mov_to_mem64(
                    MemRef {
                        reg: HEAP_PTR,
                        offset: Offset::Constant(0),
                    },
                    Arg64::Unsigned(imms.len() as u64)
                ));

                // Compile each element and move the value to the heap
                // Each element lives at [R15 + 8 * (i + 1)] since the heap grows down
                for (i, elt) in imms.iter().enumerate() {
                    is.extend(mov_to_mem64(
                        MemRef {
                            reg: HEAP_PTR,
                            offset: Offset::Constant((8 * (i + 1)) as i32)
                        },
                        compile_imm(elt, &env)
                    ));
                }

                // Tag the array location
                is.push(Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(HEAP_PTR))));
                is.push(Instr::Add(BinArgs::ToReg(Reg::Rax, Arg32::Unsigned(1))));

                // Adjust the heap pointer
                is.push(Instr::Add(BinArgs::ToReg(HEAP_PTR, Arg32::Unsigned((8 * (imms.len() + 1)) as u32))));

                is
            }

            Prim::Length => {
                let mut is = array_test(&imms[0], &env);
                is.extend(vec![
                    // If we're not indexing into an array, jump to error and terminate
                    Instr::Jne(JmpArg::Label("length_err".to_string())),

                    // Since the first word in the array is the length, we can access the memory
                    // location, retrieve that, and convert into the snake value
                    Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Reg(Reg::Rax))),
                    Instr::Sub(BinArgs::ToReg(SCRATCH, Arg32::Signed(1))),
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Mem(
                        MemRef {
                            reg: SCRATCH,
                            offset: Offset::Constant(0),
                        }
                    ))),
                    Instr::Shl(BinArgs::ToReg(Reg::Rax, Arg32::Signed(1)))
                ]);
                is
            }

            Prim::IsArray => {
                let false_label = format!("isarray_false#{}", tag);
                let done_label = format!("isarray_done#{}", tag);
                let mut is = array_test(&imms[0], &env);
                is.extend(vec![
                    Instr::Jne(JmpArg::Label(false_label.clone())),
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_TRUE))),
                    Instr::Jmp(JmpArg::Label(done_label.clone())),
                    Instr::Label(false_label),
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_FALSE))),
                    Instr::Label(done_label)
                ]);
                is
            }

            Prim::ArrayGet | Prim::ArraySet => {
                let array = &imms[0];
                let index = &imms[1];
                let mut is = array_test(array, &env);
                is.extend(
                    vec![
                        // If we are not accessing an array, then exit with an error
                        Instr::Jne(JmpArg::Label("nonarr_err".to_string())),

                        // R9 has the untagged address of the array on the heap
                        Instr::Mov(MovArgs::ToReg(SCRATCH_ARRAY, Arg64::Reg(Reg::Rax))),
                        Instr::Sub(BinArgs::ToReg(SCRATCH_ARRAY, Arg32::Signed(1))),

                        // R10 has the index i
                        // (1) i must be a number
                        Instr::Mov(MovArgs::ToReg(SCRATCH_INDEX, compile_imm(index, &env))),
                        Instr::Test(BinArgs::ToReg(SCRATCH_INDEX, Arg32::Signed(1))),
                        Instr::Jnz(JmpArg::Label("index_err".to_string())),

                        // (2) 0 <= i < n
                        Instr::Cmp(BinArgs::ToReg(SCRATCH_INDEX, Arg32::Signed(0))),
                        Instr::Jl(JmpArg::Label("segfault_err".to_string())),
                        Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Mem(MemRef {
                            reg: SCRATCH_ARRAY,
                            offset: Offset::Constant(0)
                        }))),
                        Instr::Shl(BinArgs::ToReg(Reg::Rax, Arg32::Signed(1))),
                        Instr::Cmp(BinArgs::ToReg(SCRATCH_INDEX, Arg32::Reg(Reg::Rax))),
                        Instr::Jge(JmpArg::Label("segfault_err".to_string())),

                        // R10 should have &arr[i] == (arr + 8 * (i + 1))
                        Instr::Shr(BinArgs::ToReg(SCRATCH_INDEX, Arg32::Signed(1))),
                        Instr::Shl(BinArgs::ToReg(SCRATCH_INDEX, Arg32::Signed(3))),
                        Instr::Add(BinArgs::ToReg(SCRATCH_INDEX, Arg32::Signed(8))),
                        Instr::Add(BinArgs::ToReg(SCRATCH_INDEX, Arg32::Reg(SCRATCH_ARRAY))),

                        // For both, we should have the address of arr[i]
                        Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(SCRATCH_INDEX))),
                    ]
                );

                if *op == Prim::ArraySet {
                    let value = &imms[2];
                    is.extend(
                        vec![
                            Instr::Mov(MovArgs::ToReg(SCRATCH_VALUE, compile_imm(value, &env))),
                            Instr::Mov(MovArgs::ToMem(
                                MemRef {
                                    reg: Reg::Rax,
                                    offset: Offset::Constant(0)
                                },
                                Reg32::Reg(SCRATCH_VALUE)
                            )),

                            // R9 has the original array address off by one, so we need to tag it :)
                            Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(SCRATCH_ARRAY))),
                            Instr::Add(BinArgs::ToReg(Reg::Rax, Arg32::Signed(1))),
                        ]
                    );
                } else {
                    is.push(
                        Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Mem(
                            MemRef {
                                reg: Reg::Rax,
                                offset: Offset::Constant(0),
                            }
                        )))
                    );
                }

                is
            }

            Prim::IsFun => {
                let mut is = closure_test(&imms[0], &env);
                let false_label = format!("isfun_false#{}", tag);
                let done_label = format!("isfun_done#{}", tag);
                is.extend(vec![
                    Instr::Jne(JmpArg::Label(false_label.clone())),
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_TRUE))),
                    Instr::Jmp(JmpArg::Label(done_label.clone())),
                    Instr::Label(false_label.clone()),
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Unsigned(SNAKE_FALSE))),
                    Instr::Label(done_label.clone())
                ]);
                is
            }

            Prim::GetCode => {
                vec![
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, compile_imm(&imms[0], &env))),
                    Instr::Mov(MovArgs::ToReg(
                        Reg::Rax,
                        Arg64::Mem(MemRef {
                            reg: Reg::Rax,
                            offset: Offset::Constant(8) // might be 8
                        })
                    ))
                ]
            }

            Prim::GetEnv => {
                vec![
                    Instr::Mov(MovArgs::ToReg(Reg::Rax, compile_imm(&imms[0], &env))),
                    Instr::Mov(MovArgs::ToReg(
                        Reg::Rax,
                        Arg64::Mem(MemRef {
                            reg: Reg::Rax,
                            offset: Offset::Constant(16)
                        })
                    ))
                ]
            }

            Prim::CheckArityAndUntag(size) => {
                if let ImmExp::Var(closure_name) = &imms[0] {
                    env.push(closure_name);
                    let mut is = closure_test(&imms[0], &env);
                    is.extend(vec![
                        // Check that it's a closure
                        Instr::Jne(JmpArg::Label("nonfunction_err".to_string())),

                        // Untag
                        Instr::Sub(BinArgs::ToReg(SCRATCH, Arg32::Unsigned(3))),
                        Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(SCRATCH))),

                        // Check arity
                        Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Mem(MemRef {
                            reg: SCRATCH,
                            offset: Offset::Constant(0)
                        }))),
                        Instr::Mov(MovArgs::ToReg(SCRATCH_VALUE, Arg64::Unsigned(*size as u64))),
                        Instr::Cmp(BinArgs::ToReg(SCRATCH, Arg32::Reg(SCRATCH_VALUE))),
                        Instr::Jne(JmpArg::Label("arity_err".to_string())),
                    ]);
                    is
                } else {
                    panic!("Closure should be a variable...")
                }
            }

            _ => panic!("NYI: NON-HEXHAUSTIVE CASING")
        },

        SeqExp::Let {
            var,
            bound_exp,
            body,
            ..
        } => {
            let mut is = compile_with_env(bound_exp, env.clone());
            let var_offset = env.push(var);
            is.push(Instr::Mov(MovArgs::ToMem(
                MemRef {
                    reg: Reg::Rsp,
                    offset: Offset::Constant(var_offset),
                },
                Reg32::Reg(Reg::Rax),
            )));
            is.extend(compile_with_env(body, env));
            is
        }

        SeqExp::If {
            cond,
            thn,
            els,
            ann: tag,
        } => {
            let arg = compile_imm(cond, &env);
            let thn_is = compile_with_env(thn, env.clone());
            let els_is = compile_with_env(els, env);

            let else_lab = format!("else#{}", tag);
            let done_lab = format!("done#{}", tag);

            let mut is = vec![];

            is.extend(bool_test(arg.clone(), "if_err"));

            is.extend(vec![
                Instr::Mov(MovArgs::ToReg(Reg::Rax, arg)), // mov the condition to rax
                Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Unsigned(SNAKE_TRUE))),
                Instr::Cmp(BinArgs::ToReg(Reg::Rax, Arg32::Reg(SCRATCH))), // test cond == TRUE
                Instr::Jne(JmpArg::Label(else_lab.clone())), // jump to else if the cond is 0
                // otherwise we're in the tru branch
            ]);
            is.extend(thn_is); // then branch
            is.push(Instr::Jmp(JmpArg::Label(done_lab.clone()))); // skip over the else branch
            is.push(Instr::Label(else_lab)); // else branch
            is.extend(els_is);
            is.push(Instr::Label(done_lab));
            is
        }

        SeqExp::FunDefs { decls, body, ann } => {
            let mut is = vec![];
            for decl in decls {
                env.push_fun(&decl.name);
            }
            is.extend(compile_with_env(body, env.clone()));
            let pass_label = format!("skip_local_fundefs#{}", ann);
            is.push(Instr::Jmp(JmpArg::Label(pass_label.clone())));
            for decl in decls {
                let mut local_env = env.clone();
                is.push(Instr::Label(user_fun_to_label(&decl.name)));
                for param in &decl.parameters {
                    local_env.push(param);
                }
                is.extend(compile_with_env(&decl.body, local_env));
                is.push(Instr::Ret);
            }
            is.push(Instr::Label(pass_label));
            is
        }

        SeqExp::InternalTailCall(f, args, _) => {
            let base_arg_offset = env.lookup_fun_offset(f).expect("foo");
            let mut is = vec![];
            // An internal tail call
            for (i, arg) in args.iter().enumerate() {
                is.extend(mov_to_mem64(
                    MemRef {
                        reg: Reg::Rsp,
                        offset: Offset::Constant(-8 * (1 + base_arg_offset + i32::try_from(i).unwrap())),
                    },
                    compile_imm(arg, &env),
                ))
            }
            is.push(Instr::Jmp(JmpArg::Label(user_fun_to_label(f))));
            is
        }

        SeqExp::ExternalCall {
            fun: f,
            args,
            is_tail,
            ..
        } => {
            let f = match f {
                VarOrLabel::Label(s) => s,
                VarOrLabel::Var(s) => s
            };
            let mut is = vec![];
            if *is_tail {
                // Assumption: we made temporaries for all of these so
                // if we mov them in order it's no problem
                for (i, arg) in args.iter().enumerate() {
                    is.extend(mov_to_mem64(
                        MemRef {
                            reg: Reg::Rsp,
                            offset: Offset::Constant(-8 * (1 + i32::try_from(i).unwrap())),
                        },
                        compile_imm(arg, &env),
                    ))
                }
                is.push(Instr::Jmp(JmpArg::Label(user_fun_to_label(f))));
                is
            } else {
                // say l is the num locals
                // rsp is the base.
                // last local is at [RSP - 8 * l]
                //
                // 0th arg is at
                // mov [rax - (8 * (l + 2 + 0))]
                let l = round_up_odd(env.size);

                is.push(Instr::Comment(format!("num locals: {}", l)));
                for (i, arg) in args.iter().enumerate() {
                    is.extend(mov_to_mem64(
                        MemRef {
                            reg: Reg::Rsp,
                            offset: Offset::Constant(-8 * (l + 2 + i32::try_from(i).unwrap())),
                        },
                        compile_imm(arg, &env),
                    ));
                }
                is.push(Instr::Sub(BinArgs::ToReg(Reg::Rsp, Arg32::Signed(8 * l))));
                is.push(Instr::Call(JmpArg::Label(user_fun_to_label(f))));
                is.push(Instr::Add(BinArgs::ToReg(Reg::Rsp, Arg32::Signed(8 * l))));
                is
            }
        }

        SeqExp::MakeClosure {
            arity,
            label,
            env: closure_env,
            ann
        } => {
            vec![
                // Get Heap Pointer
                Instr::Mov(MovArgs::ToReg(Reg::Rax, Arg64::Reg(HEAP_PTR))),

                // Arity at [0]
                Instr::Mov(MovArgs::ToReg(SCRATCH, Arg64::Unsigned(*arity as u64))),
                Instr::Mov(MovArgs::ToMem(
                    MemRef {
                        reg: Reg::Rax,
                        offset: Offset::Constant(0),
                    },
                    Reg32::Reg(SCRATCH)
                )),

                // Function Pointer
                Instr::RelativeLoadAddress(
                    SCRATCH_ARRAY,
                    user_fun_to_label(label)
                ),
                Instr::Mov(MovArgs::ToMem(
                    MemRef {
                        reg: Reg::Rax,
                        offset: Offset::Constant(8)
                    },
                    Reg32::Reg(SCRATCH_ARRAY)
                )),

                // Env array
                Instr::RelativeLoadAddress(
                    SCRATCH_ARRAY,
                    if let ImmExp::Var(name) = closure_env {
                        format!("{}", env.lookup_local_offset(name).unwrap_or_else(
                            || panic!("Closure name expected to be in env...")
                        ))
                    } else {
                        panic!("closure env is not a variable")
                    }
                ),
                Instr::Mov(MovArgs::ToMem(
                    MemRef {
                        reg: Reg::Rax,
                        offset: Offset::Constant(16)
                    },
                    Reg32::Reg(SCRATCH_ARRAY)
                )),

                // Tag the closure
                Instr::Add(BinArgs::ToReg(Reg::Rax, Arg32::Unsigned(3))),

                // Bump Heap Pointer by 24
                Instr::Add(BinArgs::ToReg(HEAP_PTR, Arg32::Unsigned(24)))
            ]
        }

        SeqExp::Semicolon { .. } => {
            unreachable!("Semicolons should already be desugared to lets...")
        }
    }
}

fn compile_fun(name: &str, params: &[String], body: &SeqExp<u32>) -> Vec<Instr> {
    let mut is = vec![Instr::Label(String::from(name))];
    let mut locals = CodeGenEnv::new(vec![]);
    for param_name in params {
        locals.push(param_name);
    }
    is.extend(compile_with_env(body, locals));
    is.push(Instr::Ret);
    is
}

fn remove_tag<Span>(p: &Exp<Span>) -> Exp<()> {
    match p {
        Exp::Num(n, ..) => Exp::Num(*n, ()),
        Exp::Bool(b, ..) => Exp::Bool(*b, ()),
        Exp::Var(x, ..) => Exp::Var(x.clone(), ()),
        Exp::Prim(op, exps, ..) => Exp::Prim(
            op.clone(),
            exps.iter()
                .map(|e| Box::new(remove_tag(e)))
                .collect(),
            (),
        ),
        Exp::Let {
            bindings,
            body,
            ..
        } => Exp::Let {
            bindings: bindings
                .iter()
                .map(|(x, e)| (x.clone(), remove_tag(e)))
                .collect(),
            body: Box::new(remove_tag(body)),
            ann: (),
        },
        Exp::If {
            cond,
            thn,
            els,
            ..
        } => Exp::If {
            cond: Box::new(remove_tag(cond)),
            thn: Box::new(remove_tag(thn)),
            els: Box::new(remove_tag(els)),
            ann: (),
        },
        Exp::Semicolon { e1, e2, .. } => Exp::Semicolon {
            e1: Box::new(remove_tag(e1)),
            e2: Box::new(remove_tag(e2)),
            ann: (),
        },
        Exp::FunDefs {
            decls,
            body,
            ..
        } => Exp::FunDefs {
            decls: decls
                .iter()
                .map(|d| FunDecl {
                    name: d.name.clone(),
                    parameters: d.parameters.clone(),
                    body: remove_tag(&d.body),
                    ann: (),
                })
                .collect(),
            body: Box::new(remove_tag(body)),
            ann: (),
        },
        Exp::Lambda {
            parameters,
            body,
            ..
        } => Exp::Lambda {
            parameters: parameters.clone(),
            body: Box::new(remove_tag(body)),
            ann: (),
        },
        Exp::MakeClosure {
            arity,
            label,
            env,
            ..
        } => Exp::MakeClosure {
            arity: *arity,
            label: label.clone(),
            env: Box::new(remove_tag(env)),
            ann: (),
        },
        Exp::Call(f, args, ..) => Exp::Call(
            Box::new(remove_tag(f)),
            args.iter().map(|e| remove_tag(e)).collect(),
            (),
        ),
        Exp::InternalTailCall(f, args, ..) => Exp::InternalTailCall(
            f.clone(),
            args.iter().map(|e| remove_tag(e)).collect(),
            (),
        ),
        Exp::ExternalCall {
            fun,
            args,
            is_tail,
            ..
        } => Exp::ExternalCall {
            fun: fun.clone(),
            args: args.iter().map(|e| remove_tag(e)).collect(),
            is_tail: *is_tail,
            ann: (),
        },
        Exp::ClosureCall(f, args, ..) => Exp::ClosureCall(
            Box::new(remove_tag(f)),
            args.iter().map(|e| remove_tag(e)).collect(),
            (),
        ),
        Exp::DirectCall(f, args, ..) => Exp::DirectCall(
            f.to_string(),
            args.iter().map(|e| remove_tag(e)).collect(),
            (),
        ),
    }
}

pub fn compile_to_string<Span>(p: &SurfProg<Span>) -> Result<String, CompileErr<Span>>
    where
        Span: Clone,
{
    // Print the expression
    // println!("{:#?}", remove_tag(p));

    // first check for errors
    check_prog(p)?;

    // then give all the variables unique names
    let uniq_p = uniquify(&tag_exp(p));

    // lift definitions to the top level
    let (defs, main) = lambda_lift(&uniq_p);
    // eprintln!("lifted: {:#?}\n{:#?}", defs, main);

    // tag the lifted expressions
    let (t_defs, t_main) = tag_prog(&defs, &main);

    // then sequentialize
    let seq_p = tag_sprog(&seq_prog(&t_defs, &t_main));
    // eprintln!("sequential prog:\n{:#?}", seq_p);

    // finally codegen
    let header = "\
        section .data
        HEAP: times 1024 dq 0
        section .text
        global start_here
        extern print_snake_val
        extern snake_error
start_here:
        push r15             ; save the original value to r15
        sub rsp, 8           ; padding to ensure the correct alignment
        lea r15, [rel HEAP]  ; load the address of the HEAP into r15 using rip-relative address
        call main            ; call into the actual code for the main expr
        add rsp, 8           ; remove the padding
        pop r15              ; restore the original to r15
        ret
overflow_err:
        mov rdi, 0
        call snake_error
arith_err:
        mov rdi, 1
        call snake_error
cmp_err:
        mov rdi, 2
        call snake_error
log_err:
        mov rdi, 3
        call snake_error
if_err:
        mov rdi, 4
        call snake_error
segfault_err:
        mov rdi, 5
        call snake_error
nonarr_err:
        mov rdi, 6
        call snake_error
index_err:
        mov rdi, 7
        call snake_error
length_err:
        mov rdi, 8
        call snake_error
nonfunction_err:
        mov rdi, 9
        call snake_error
arity_err:
        mov rdi, 10
        call snake_error
 ";
    let mut buf = String::from(header);
    buf.push_str(&instrs_to_string(&compile_fun("main", &[], &seq_p.main)));
    for d in seq_p.funs.iter() {
        buf.push_str(&instrs_to_string(&compile_fun(
            &user_fun_to_label(&d.name),
            &d.parameters,
            &d.body,
        )))
    }
    eprintln!("{}", buf.clone());
    Ok(buf)
}

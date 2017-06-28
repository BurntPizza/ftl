use std::str::FromStr;
use ::*;
extern crate lalrpop_util as __lalrpop_util;

mod __parse__program {
    #![allow(non_snake_case, non_camel_case_types, unused_mut, unused_variables, unused_imports)]

    use std::str::FromStr;
    use ::*;
    extern crate lalrpop_util as __lalrpop_util;
    #[allow(dead_code)]
    pub enum __Symbol<'input> {
        Term_22_28_22(&'input str),
        Term_22_29_22(&'input str),
        Term_22_3a_22(&'input str),
        Term_22_3b_22(&'input str),
        Term_22break_3b_22(&'input str),
        Term_22case_20_22(&'input str),
        Term_22print_20_22(&'input str),
        Term_22switch_20_22(&'input str),
        Term_22_7b_22(&'input str),
        Term_22_7d_22(&'input str),
        Termr_23_22_2d_3f_5b0_2d9_5d_2b_22_23(&'input str),
        Termr_23_22_5ba_2dzA_2dZ___5d_5ba_2dzA_2dZ__0_2d9_5d_2a_22_23(&'input str),
        Termr_23_22read_5c_5c_28_5c_5c_29_22_23(&'input str),
        Nt____program(Program),
        Ntcase(Case),
        Ntcase_2a(::std::vec::Vec<Case>),
        Ntcase_2b(::std::vec::Vec<Case>),
        Ntcases(::std::vec::Vec<Case>),
        Ntexpr(Expr),
        Ntident(String),
        Ntnum(i64),
        Ntprogram(Program),
        Ntstatement(Statement),
        Ntstatement_2a(::std::vec::Vec<Statement>),
        Ntstatement_2b(::std::vec::Vec<Statement>),
        Ntstmt__list(::std::vec::Vec<Statement>),
        Ntswitch(Switch),
    }
    const __ACTION: &'static [i32] = &[
        // State 0
        0, 0, 0, 0, 5, 0, 6, 7, 8, 0, 0, 0, 0,
        // State 1
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        // State 2
        -14, -14, -14, -14, -14, -14, -14, -14, -14, -14, -14, -14, -14,
        // State 3
        -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
        // State 4
        -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
        // State 5
        11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 13,
        // State 6
        11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 13,
        // State 7
        0, 0, 0, 0, 5, 0, 6, 7, 8, -23, 0, 0, 0,
        // State 8
        0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 9
        -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
        // State 10
        11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 13,
        // State 11
        -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13,
        // State 12
        -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9,
        // State 13
        0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0,
        // State 14
        -21, -21, -21, -21, -21, -21, -21, -21, -21, -21, -21, -21, -21,
        // State 15
        0, 0, 0, 0, 5, 0, 6, 7, 8, -24, 0, 0, 0,
        // State 16
        0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0,
        // State 17
        -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15,
        // State 18
        0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 19
        0, 0, 0, 0, 0, 27, 0, 0, 0, -7, 0, 0, 0,
        // State 20
        -22, -22, -22, -22, -22, -22, -22, -22, -22, -22, -22, -22, -22,
        // State 21
        -18, -18, -18, -18, -18, -18, -18, -18, -18, -18, -18, -18, -18,
        // State 22
        -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
        // State 23
        -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
        // State 24
        0, 0, 0, 0, 0, 27, 0, 0, 0, -8, 0, 0, 0,
        // State 25
        0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0,
        // State 26
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0,
        // State 27
        -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6,
        // State 28
        -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25,
        // State 29
        0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 30
        0, 0, 0, 0, 5, 0, 6, 7, 8, 0, 0, 0, 0,
        // State 31
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    ];
    const __EOF_ACTION: &'static [i32] = &[
        0,
        -1,
        -14,
        -17,
        -16,
        0,
        0,
        0,
        0,
        -10,
        0,
        -13,
        -9,
        0,
        -21,
        0,
        0,
        -15,
        0,
        0,
        -22,
        -18,
        -11,
        -5,
        0,
        0,
        0,
        -6,
        -25,
        0,
        0,
        -2,
    ];
    const __GOTO: &'static [i32] = &[
        // State 0
        0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 4,
        // State 1
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 2
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 3
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 4
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 5
        0, 0, 0, 0, 0, 9, 0, 10, 0, 0, 0, 0, 0, 0,
        // State 6
        0, 0, 0, 0, 0, 14, 0, 10, 0, 0, 0, 0, 0, 0,
        // State 7
        0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 16, 17, 4,
        // State 8
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 9
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 10
        0, 0, 0, 0, 0, 19, 0, 10, 0, 0, 0, 0, 0, 0,
        // State 11
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 12
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 13
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 14
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 15
        0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 4,
        // State 16
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 17
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 18
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 19
        0, 24, 0, 25, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 20
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 21
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 22
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 23
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 24
        0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 25
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 26
        0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0,
        // State 27
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 28
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 29
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 30
        0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 4,
        // State 31
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    fn __expected_tokens(__state: usize) -> Vec<::std::string::String> {
        const __TERMINAL: &'static [&'static str] = &[
            r###""(""###,
            r###"")""###,
            r###"":""###,
            r###"";""###,
            r###""break;""###,
            r###""case ""###,
            r###""print ""###,
            r###""switch ""###,
            r###""{""###,
            r###""}""###,
            r###"r#"-?[0-9]+"#"###,
            r###"r#"[a-zA-Z_][a-zA-Z_0-9]*"#"###,
            r###"r#"read\\(\\)"#"###,
        ];
        __ACTION[(__state * 13)..].iter().zip(__TERMINAL).filter_map(|(&state, terminal)| {
            if state == 0 {
                None
            } else {
                Some(terminal.to_string())
            }
        }).collect()
    }
    pub fn parse_program<
        'input,
    >(
        input: &'input str,
    ) -> Result<Program, __lalrpop_util::ParseError<usize, (usize, &'input str), ()>>
    {
        let mut __tokens = super::__intern_token::__Matcher::new(input);
        let mut __states = vec![0_i32];
        let mut __symbols = vec![];
        let mut __integer;
        let mut __lookahead;
        let mut __last_location = Default::default();
        '__shift: loop {
            __lookahead = match __tokens.next() {
                Some(Ok(v)) => v,
                None => break '__shift,
                Some(Err(e)) => return Err(e),
            };
            __last_location = __lookahead.2.clone();
            __integer = match __lookahead.1 {
                (3, _) if true => 0,
                (4, _) if true => 1,
                (5, _) if true => 2,
                (6, _) if true => 3,
                (7, _) if true => 4,
                (8, _) if true => 5,
                (9, _) if true => 6,
                (10, _) if true => 7,
                (11, _) if true => 8,
                (12, _) if true => 9,
                (0, _) if true => 10,
                (1, _) if true => 11,
                (2, _) if true => 12,
                _ => {
                    let __state = *__states.last().unwrap() as usize;
                    let __error = __lalrpop_util::ParseError::UnrecognizedToken {
                        token: Some(__lookahead),
                        expected: __expected_tokens(__state),
                    };
                    return Err(__error);
                }
            };
            '__inner: loop {
                let __state = *__states.last().unwrap() as usize;
                let __action = __ACTION[__state * 13 + __integer];
                if __action > 0 {
                    let __symbol = match __integer {
                        0 => match __lookahead.1 {
                            (3, __tok0) => __Symbol::Term_22_28_22((__tok0)),
                            _ => unreachable!(),
                        },
                        1 => match __lookahead.1 {
                            (4, __tok0) => __Symbol::Term_22_29_22((__tok0)),
                            _ => unreachable!(),
                        },
                        2 => match __lookahead.1 {
                            (5, __tok0) => __Symbol::Term_22_3a_22((__tok0)),
                            _ => unreachable!(),
                        },
                        3 => match __lookahead.1 {
                            (6, __tok0) => __Symbol::Term_22_3b_22((__tok0)),
                            _ => unreachable!(),
                        },
                        4 => match __lookahead.1 {
                            (7, __tok0) => __Symbol::Term_22break_3b_22((__tok0)),
                            _ => unreachable!(),
                        },
                        5 => match __lookahead.1 {
                            (8, __tok0) => __Symbol::Term_22case_20_22((__tok0)),
                            _ => unreachable!(),
                        },
                        6 => match __lookahead.1 {
                            (9, __tok0) => __Symbol::Term_22print_20_22((__tok0)),
                            _ => unreachable!(),
                        },
                        7 => match __lookahead.1 {
                            (10, __tok0) => __Symbol::Term_22switch_20_22((__tok0)),
                            _ => unreachable!(),
                        },
                        8 => match __lookahead.1 {
                            (11, __tok0) => __Symbol::Term_22_7b_22((__tok0)),
                            _ => unreachable!(),
                        },
                        9 => match __lookahead.1 {
                            (12, __tok0) => __Symbol::Term_22_7d_22((__tok0)),
                            _ => unreachable!(),
                        },
                        10 => match __lookahead.1 {
                            (0, __tok0) => __Symbol::Termr_23_22_2d_3f_5b0_2d9_5d_2b_22_23((__tok0)),
                            _ => unreachable!(),
                        },
                        11 => match __lookahead.1 {
                            (1, __tok0) => __Symbol::Termr_23_22_5ba_2dzA_2dZ___5d_5ba_2dzA_2dZ__0_2d9_5d_2a_22_23((__tok0)),
                            _ => unreachable!(),
                        },
                        12 => match __lookahead.1 {
                            (2, __tok0) => __Symbol::Termr_23_22read_5c_5c_28_5c_5c_29_22_23((__tok0)),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    __states.push(__action - 1);
                    __symbols.push((__lookahead.0, __symbol, __lookahead.2));
                    continue '__shift;
                } else if __action < 0 {
                    if let Some(r) = __reduce(input, __action, Some(&__lookahead.0), &mut __states, &mut __symbols, ::std::marker::PhantomData::<()>) {
                        return r;
                    }
                } else {
                    let __state = *__states.last().unwrap() as usize;
                    let __error = __lalrpop_util::ParseError::UnrecognizedToken {
                        token: Some(__lookahead),
                        expected: __expected_tokens(__state),
                    };
                    return Err(__error)
                }
            }
        }
        loop {
            let __state = *__states.last().unwrap() as usize;
            let __action = __EOF_ACTION[__state];
            if __action < 0 {
                if let Some(r) = __reduce(input, __action, None, &mut __states, &mut __symbols, ::std::marker::PhantomData::<()>) {
                    return r;
                }
            } else {
                let __state = *__states.last().unwrap() as usize;
                let __error = __lalrpop_util::ParseError::UnrecognizedToken {
                    token: None,
                    expected: __expected_tokens(__state),
                };
                return Err(__error);
            }
        }
    }
    pub fn __reduce<
        'input,
    >(
        input: &'input str,
        __action: i32,
        __lookahead_start: Option<&usize>,
        __states: &mut ::std::vec::Vec<i32>,
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>,
        _: ::std::marker::PhantomData<()>,
    ) -> Option<Result<Program,__lalrpop_util::ParseError<usize, (usize, &'input str), ()>>>
    {
        let __nonterminal = match -__action {
            1 => {
                // __program = program => ActionFn(0);
                let __sym0 = __pop_Ntprogram(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action0::<>(input, __sym0);
                return Some(Ok(__nt));
            }
            2 => {
                // case = "case ", num, ":", statement => ActionFn(7);
                let __sym3 = __pop_Ntstatement(__symbols);
                let __sym2 = __pop_Term_22_3a_22(__symbols);
                let __sym1 = __pop_Ntnum(__symbols);
                let __sym0 = __pop_Term_22case_20_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym3.2.clone();
                let __nt = super::__action7::<>(input, __sym0, __sym1, __sym2, __sym3);
                let __states_len = __states.len();
                __states.truncate(__states_len - 4);
                __symbols.push((__start, __Symbol::Ntcase(__nt), __end));
                1
            }
            3 => {
                // case* =  => ActionFn(15);
                let __start = __symbols.last().map(|s| s.2.clone()).unwrap_or_default();
                let __end = __lookahead_start.cloned().unwrap_or_else(|| __start.clone());
                let __nt = super::__action15::<>(input, &__start, &__end);
                let __states_len = __states.len();
                __states.truncate(__states_len - 0);
                __symbols.push((__start, __Symbol::Ntcase_2a(__nt), __end));
                2
            }
            4 => {
                // case* = case+ => ActionFn(16);
                let __sym0 = __pop_Ntcase_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action16::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntcase_2a(__nt), __end));
                2
            }
            5 => {
                // case+ = case => ActionFn(21);
                let __sym0 = __pop_Ntcase(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action21::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntcase_2b(__nt), __end));
                3
            }
            6 => {
                // case+ = case+, case => ActionFn(22);
                let __sym1 = __pop_Ntcase(__symbols);
                let __sym0 = __pop_Ntcase_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym1.2.clone();
                let __nt = super::__action22::<>(input, __sym0, __sym1);
                let __states_len = __states.len();
                __states.truncate(__states_len - 2);
                __symbols.push((__start, __Symbol::Ntcase_2b(__nt), __end));
                3
            }
            7 => {
                // cases =  => ActionFn(23);
                let __start = __symbols.last().map(|s| s.2.clone()).unwrap_or_default();
                let __end = __lookahead_start.cloned().unwrap_or_else(|| __start.clone());
                let __nt = super::__action23::<>(input, &__start, &__end);
                let __states_len = __states.len();
                __states.truncate(__states_len - 0);
                __symbols.push((__start, __Symbol::Ntcases(__nt), __end));
                4
            }
            8 => {
                // cases = case+ => ActionFn(24);
                let __sym0 = __pop_Ntcase_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action24::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntcases(__nt), __end));
                4
            }
            9 => {
                // expr = r#"read\\(\\)"# => ActionFn(8);
                let __sym0 = __pop_Termr_23_22read_5c_5c_28_5c_5c_29_22_23(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action8::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntexpr(__nt), __end));
                5
            }
            10 => {
                // expr = num => ActionFn(9);
                let __sym0 = __pop_Ntnum(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action9::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntexpr(__nt), __end));
                5
            }
            11 => {
                // expr = "(", expr, ")" => ActionFn(10);
                let __sym2 = __pop_Term_22_29_22(__symbols);
                let __sym1 = __pop_Ntexpr(__symbols);
                let __sym0 = __pop_Term_22_28_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action10::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::Ntexpr(__nt), __end));
                5
            }
            12 => {
                // ident = r#"[a-zA-Z_][a-zA-Z_0-9]*"# => ActionFn(14);
                let __sym0 = __pop_Termr_23_22_5ba_2dzA_2dZ___5d_5ba_2dzA_2dZ__0_2d9_5d_2a_22_23(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action14::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntident(__nt), __end));
                6
            }
            13 => {
                // num = r#"-?[0-9]+"# => ActionFn(13);
                let __sym0 = __pop_Termr_23_22_2d_3f_5b0_2d9_5d_2b_22_23(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action13::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntnum(__nt), __end));
                7
            }
            14 => {
                // program = statement => ActionFn(1);
                let __sym0 = __pop_Ntstatement(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action1::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntprogram(__nt), __end));
                8
            }
            15 => {
                // statement = "print ", expr, ";" => ActionFn(2);
                let __sym2 = __pop_Term_22_3b_22(__symbols);
                let __sym1 = __pop_Ntexpr(__symbols);
                let __sym0 = __pop_Term_22print_20_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action2::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::Ntstatement(__nt), __end));
                9
            }
            16 => {
                // statement = "break;" => ActionFn(3);
                let __sym0 = __pop_Term_22break_3b_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action3::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntstatement(__nt), __end));
                9
            }
            17 => {
                // statement = switch => ActionFn(4);
                let __sym0 = __pop_Ntswitch(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action4::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntstatement(__nt), __end));
                9
            }
            18 => {
                // statement = "{", stmt_list, "}" => ActionFn(5);
                let __sym2 = __pop_Term_22_7d_22(__symbols);
                let __sym1 = __pop_Ntstmt__list(__symbols);
                let __sym0 = __pop_Term_22_7b_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action5::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::Ntstatement(__nt), __end));
                9
            }
            19 => {
                // statement* =  => ActionFn(17);
                let __start = __symbols.last().map(|s| s.2.clone()).unwrap_or_default();
                let __end = __lookahead_start.cloned().unwrap_or_else(|| __start.clone());
                let __nt = super::__action17::<>(input, &__start, &__end);
                let __states_len = __states.len();
                __states.truncate(__states_len - 0);
                __symbols.push((__start, __Symbol::Ntstatement_2a(__nt), __end));
                10
            }
            20 => {
                // statement* = statement+ => ActionFn(18);
                let __sym0 = __pop_Ntstatement_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action18::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntstatement_2a(__nt), __end));
                10
            }
            21 => {
                // statement+ = statement => ActionFn(19);
                let __sym0 = __pop_Ntstatement(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action19::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntstatement_2b(__nt), __end));
                11
            }
            22 => {
                // statement+ = statement+, statement => ActionFn(20);
                let __sym1 = __pop_Ntstatement(__symbols);
                let __sym0 = __pop_Ntstatement_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym1.2.clone();
                let __nt = super::__action20::<>(input, __sym0, __sym1);
                let __states_len = __states.len();
                __states.truncate(__states_len - 2);
                __symbols.push((__start, __Symbol::Ntstatement_2b(__nt), __end));
                11
            }
            23 => {
                // stmt_list =  => ActionFn(25);
                let __start = __symbols.last().map(|s| s.2.clone()).unwrap_or_default();
                let __end = __lookahead_start.cloned().unwrap_or_else(|| __start.clone());
                let __nt = super::__action25::<>(input, &__start, &__end);
                let __states_len = __states.len();
                __states.truncate(__states_len - 0);
                __symbols.push((__start, __Symbol::Ntstmt__list(__nt), __end));
                12
            }
            24 => {
                // stmt_list = statement+ => ActionFn(26);
                let __sym0 = __pop_Ntstatement_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action26::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::Ntstmt__list(__nt), __end));
                12
            }
            25 => {
                // switch = "switch ", expr, "{", cases, "}" => ActionFn(6);
                let __sym4 = __pop_Term_22_7d_22(__symbols);
                let __sym3 = __pop_Ntcases(__symbols);
                let __sym2 = __pop_Term_22_7b_22(__symbols);
                let __sym1 = __pop_Ntexpr(__symbols);
                let __sym0 = __pop_Term_22switch_20_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym4.2.clone();
                let __nt = super::__action6::<>(input, __sym0, __sym1, __sym2, __sym3, __sym4);
                let __states_len = __states.len();
                __states.truncate(__states_len - 5);
                __symbols.push((__start, __Symbol::Ntswitch(__nt), __end));
                13
            }
            _ => panic!("invalid action code {}", __action)
        };
        let __state = *__states.last().unwrap() as usize;
        let __next_state = __GOTO[__state * 14 + __nonterminal] - 1;
        __states.push(__next_state);
        None
    }
    fn __pop_Term_22_28_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_28_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_29_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_29_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_3a_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_3a_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_3b_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_3b_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22break_3b_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22break_3b_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22case_20_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22case_20_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22print_20_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22print_20_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22switch_20_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22switch_20_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_7b_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_7b_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_7d_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_7d_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Termr_23_22_2d_3f_5b0_2d9_5d_2b_22_23<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Termr_23_22_2d_3f_5b0_2d9_5d_2b_22_23(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Termr_23_22_5ba_2dzA_2dZ___5d_5ba_2dzA_2dZ__0_2d9_5d_2a_22_23<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Termr_23_22_5ba_2dzA_2dZ___5d_5ba_2dzA_2dZ__0_2d9_5d_2a_22_23(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Termr_23_22read_5c_5c_28_5c_5c_29_22_23<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Termr_23_22read_5c_5c_28_5c_5c_29_22_23(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Nt____program<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Program, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Nt____program(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntcase<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Case, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntcase(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntcase_2a<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Case>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntcase_2a(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntcase_2b<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Case>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntcase_2b(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntcases<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Case>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntcases(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntexpr<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Expr, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntexpr(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntident<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, String, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntident(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntnum<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, i64, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntnum(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntprogram<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Program, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntprogram(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntstatement<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Statement, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntstatement(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntstatement_2a<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Statement>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntstatement_2a(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntstatement_2b<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Statement>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntstatement_2b(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntstmt__list<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Statement>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntstmt__list(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Ntswitch<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Switch, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Ntswitch(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
}
pub use self::__parse__program::parse_program;
mod __intern_token {
    #![allow(unused_imports)]
    use std::str::FromStr;
    use ::*;
    extern crate lalrpop_util as __lalrpop_util;
    extern crate regex as __regex;
    pub struct __Matcher<'input> {
        text: &'input str,
        consumed: usize,
        regex_set: __regex::RegexSet,
        regex_vec: Vec<__regex::Regex>,
    }

    impl<'input> __Matcher<'input> {
        pub fn new(s: &'input str) -> __Matcher<'input> {
            let __strs: &[&str] = &[
                "^(?u:\\-)?(?u:[0-9])+",
                "^(?u:[A-Z_-_a-z])(?u:[0-9A-Z_-_a-z])*",
                "^(?u:read\\(\\))",
                "^(?u:\\()",
                "^(?u:\\))",
                "^(?u::)",
                "^(?u:;)",
                "^(?u:break;)",
                "^(?u:case )",
                "^(?u:print )",
                "^(?u:switch )",
                "^(?u:\\{)",
                "^(?u:\\})",
            ];
            let __regex_set = __regex::RegexSet::new(__strs).unwrap();
            let __regex_vec = vec![
                __regex::Regex::new("^(?u:\\-)?(?u:[0-9])+").unwrap(),
                __regex::Regex::new("^(?u:[A-Z_-_a-z])(?u:[0-9A-Z_-_a-z])*").unwrap(),
                __regex::Regex::new("^(?u:read\\(\\))").unwrap(),
                __regex::Regex::new("^(?u:\\()").unwrap(),
                __regex::Regex::new("^(?u:\\))").unwrap(),
                __regex::Regex::new("^(?u::)").unwrap(),
                __regex::Regex::new("^(?u:;)").unwrap(),
                __regex::Regex::new("^(?u:break;)").unwrap(),
                __regex::Regex::new("^(?u:case )").unwrap(),
                __regex::Regex::new("^(?u:print )").unwrap(),
                __regex::Regex::new("^(?u:switch )").unwrap(),
                __regex::Regex::new("^(?u:\\{)").unwrap(),
                __regex::Regex::new("^(?u:\\})").unwrap(),
            ];
            __Matcher {
                text: s,
                consumed: 0,
                regex_set: __regex_set,
                regex_vec: __regex_vec,
            }
        }
    }

    impl<'input> Iterator for __Matcher<'input> {
        type Item = Result<(usize, (usize, &'input str), usize), __lalrpop_util::ParseError<usize,(usize, &'input str),()>>;

        fn next(&mut self) -> Option<Self::Item> {
            let __text = self.text.trim_left();
            let __whitespace = self.text.len() - __text.len();
            let __start_offset = self.consumed + __whitespace;
            if __text.is_empty() {
                self.text = __text;
                self.consumed = __start_offset;
                None
            } else {
                let __matches = self.regex_set.matches(__text);
                if !__matches.matched_any() {
                    Some(Err(__lalrpop_util::ParseError::InvalidToken {
                        location: __start_offset,
                    }))
                } else {
                    let mut __longest_match = 0;
                    let mut __index = 0;
                    for __i in 0 .. 13 {
                        if __matches.matched(__i) {
                            let __match = self.regex_vec[__i].find(__text).unwrap();
                            let __len = __match.end();
                            if __len >= __longest_match {
                                __longest_match = __len;
                                __index = __i;
                            }
                        }
                    }
                    let __result = &__text[..__longest_match];
                    let __remaining = &__text[__longest_match..];
                    let __end_offset = __start_offset + __longest_match;
                    self.text = __remaining;
                    self.consumed = __end_offset;
                    Some(Ok((__start_offset, (__index, __result), __end_offset)))
                }
            }
        }
    }
}

#[allow(unused_variables)]
fn __action0<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Program, usize),
) -> Program
{
    (__0)
}

#[allow(unused_variables)]
fn __action1<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Statement, usize),
) -> Program
{
    Program(__0)
}

#[allow(unused_variables)]
fn __action2<
    'input,
>(
    input: &'input str,
    (_, _, _): (usize, &'input str, usize),
    (_, e, _): (usize, Expr, usize),
    (_, _, _): (usize, &'input str, usize),
) -> Statement
{
    Statement::Print(e)
}

#[allow(unused_variables)]
fn __action3<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Statement
{
    Statement::Break
}

#[allow(unused_variables)]
fn __action4<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Switch, usize),
) -> Statement
{
    Statement::Switch(__0)
}

#[allow(unused_variables)]
fn __action5<
    'input,
>(
    input: &'input str,
    (_, _, _): (usize, &'input str, usize),
    (_, s, _): (usize, ::std::vec::Vec<Statement>, usize),
    (_, _, _): (usize, &'input str, usize),
) -> Statement
{
    Statement::Block(s)
}

#[allow(unused_variables)]
fn __action6<
    'input,
>(
    input: &'input str,
    (_, _, _): (usize, &'input str, usize),
    (_, arg, _): (usize, Expr, usize),
    (_, _, _): (usize, &'input str, usize),
    (_, cases, _): (usize, ::std::vec::Vec<Case>, usize),
    (_, _, _): (usize, &'input str, usize),
) -> Switch
{
    Switch { arg: Box::new(arg), cases: cases }
}

#[allow(unused_variables)]
fn __action7<
    'input,
>(
    input: &'input str,
    (_, _, _): (usize, &'input str, usize),
    (_, n, _): (usize, i64, usize),
    (_, _, _): (usize, &'input str, usize),
    (_, s, _): (usize, Statement, usize),
) -> Case
{
    Case(n, s)
}

#[allow(unused_variables)]
fn __action8<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Expr
{
    Expr::Read
}

#[allow(unused_variables)]
fn __action9<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, i64, usize),
) -> Expr
{
    Expr::I64(__0)
}

#[allow(unused_variables)]
fn __action10<
    'input,
>(
    input: &'input str,
    (_, _, _): (usize, &'input str, usize),
    (_, e, _): (usize, Expr, usize),
    (_, _, _): (usize, &'input str, usize),
) -> Expr
{
    e
}

#[allow(unused_variables)]
fn __action11<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, ::std::vec::Vec<Statement>, usize),
) -> ::std::vec::Vec<Statement>
{
    (__0)
}

#[allow(unused_variables)]
fn __action12<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, ::std::vec::Vec<Case>, usize),
) -> ::std::vec::Vec<Case>
{
    (__0)
}

#[allow(unused_variables)]
fn __action13<
    'input,
>(
    input: &'input str,
    (_, s, _): (usize, &'input str, usize),
) -> i64
{
    i64::from_str(s).unwrap()
}

#[allow(unused_variables)]
fn __action14<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> String
{
    __0.to_owned()
}

#[allow(unused_variables)]
fn __action15<
    'input,
>(
    input: &'input str,
    __lookbehind: &usize,
    __lookahead: &usize,
) -> ::std::vec::Vec<Case>
{
    vec![]
}

#[allow(unused_variables)]
fn __action16<
    'input,
>(
    input: &'input str,
    (_, v, _): (usize, ::std::vec::Vec<Case>, usize),
) -> ::std::vec::Vec<Case>
{
    v
}

#[allow(unused_variables)]
fn __action17<
    'input,
>(
    input: &'input str,
    __lookbehind: &usize,
    __lookahead: &usize,
) -> ::std::vec::Vec<Statement>
{
    vec![]
}

#[allow(unused_variables)]
fn __action18<
    'input,
>(
    input: &'input str,
    (_, v, _): (usize, ::std::vec::Vec<Statement>, usize),
) -> ::std::vec::Vec<Statement>
{
    v
}

#[allow(unused_variables)]
fn __action19<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Statement, usize),
) -> ::std::vec::Vec<Statement>
{
    vec![__0]
}

#[allow(unused_variables)]
fn __action20<
    'input,
>(
    input: &'input str,
    (_, v, _): (usize, ::std::vec::Vec<Statement>, usize),
    (_, e, _): (usize, Statement, usize),
) -> ::std::vec::Vec<Statement>
{
    { let mut v = v; v.push(e); v }
}

#[allow(unused_variables)]
fn __action21<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Case, usize),
) -> ::std::vec::Vec<Case>
{
    vec![__0]
}

#[allow(unused_variables)]
fn __action22<
    'input,
>(
    input: &'input str,
    (_, v, _): (usize, ::std::vec::Vec<Case>, usize),
    (_, e, _): (usize, Case, usize),
) -> ::std::vec::Vec<Case>
{
    { let mut v = v; v.push(e); v }
}

#[allow(unused_variables)]
fn __action23<
    'input,
>(
    input: &'input str,
    __lookbehind: &usize,
    __lookahead: &usize,
) -> ::std::vec::Vec<Case>
{
    let __start0 = __lookbehind.clone();
    let __end0 = __lookahead.clone();
    let __temp0 = __action15(
        input,
        &__start0,
        &__end0,
    );
    let __temp0 = (__start0, __temp0, __end0);
    __action12(
        input,
        __temp0,
    )
}

#[allow(unused_variables)]
fn __action24<
    'input,
>(
    input: &'input str,
    __0: (usize, ::std::vec::Vec<Case>, usize),
) -> ::std::vec::Vec<Case>
{
    let __start0 = __0.0.clone();
    let __end0 = __0.2.clone();
    let __temp0 = __action16(
        input,
        __0,
    );
    let __temp0 = (__start0, __temp0, __end0);
    __action12(
        input,
        __temp0,
    )
}

#[allow(unused_variables)]
fn __action25<
    'input,
>(
    input: &'input str,
    __lookbehind: &usize,
    __lookahead: &usize,
) -> ::std::vec::Vec<Statement>
{
    let __start0 = __lookbehind.clone();
    let __end0 = __lookahead.clone();
    let __temp0 = __action17(
        input,
        &__start0,
        &__end0,
    );
    let __temp0 = (__start0, __temp0, __end0);
    __action11(
        input,
        __temp0,
    )
}

#[allow(unused_variables)]
fn __action26<
    'input,
>(
    input: &'input str,
    __0: (usize, ::std::vec::Vec<Statement>, usize),
) -> ::std::vec::Vec<Statement>
{
    let __start0 = __0.0.clone();
    let __end0 = __0.2.clone();
    let __temp0 = __action18(
        input,
        __0,
    );
    let __temp0 = (__start0, __temp0, __end0);
    __action11(
        input,
        __temp0,
    )
}

pub trait __ToTriple<'input, > {
    type Error;
    fn to_triple(value: Self) -> Result<(usize,(usize, &'input str),usize),Self::Error>;
}

impl<'input, > __ToTriple<'input, > for (usize, (usize, &'input str), usize) {
    type Error = ();
    fn to_triple(value: Self) -> Result<(usize,(usize, &'input str),usize),()> {
        Ok(value)
    }
}
impl<'input, > __ToTriple<'input, > for Result<(usize, (usize, &'input str), usize),()> {
    type Error = ();
    fn to_triple(value: Self) -> Result<(usize,(usize, &'input str),usize),()> {
        value
    }
}

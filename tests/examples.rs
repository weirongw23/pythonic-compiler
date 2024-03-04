use snake::runner;

macro_rules! mk_test {
    ($test_name:ident, $file_name:expr, $expected_output:expr) => {
        #[test]
        fn $test_name() -> std::io::Result<()> {
            test_example_file($file_name, $expected_output)
        }
    };
}

macro_rules! mk_fail_test {
    ($test_name:ident, $file_name:expr, $expected_output:expr) => {
        #[test]
        fn $test_name() -> std::io::Result<()> {
            test_example_fail($file_name, $expected_output)
        }
    };
}

// Error-Handling: Static Errors
mk_fail_test!(
    overflow_max,
    "error::overflowmax.egg",
    "Number literal 4611686018427387904 doesn't fit into 63-bit integer"
);

mk_fail_test!(
    overflow_min,
    "error::overflowmin.egg",
    "Number literal -4611686018427387905 doesn't fit into 63-bit integer"
);

mk_fail_test!(
    undefined_function,
    "error::undef.egg",
    "Unbound variable f"
);

mk_fail_test!(
    undefined_variable,
    "error::unboundvar.egg",
    "Unbound variable y"
);

mk_fail_test!(
    duplicate_variable,
    "error::dupvar.egg",
    "Variable x defined twice"
);

mk_fail_test!(
    duplicate_arg,
    "error::duparg.egg",
    "multiple arguments named \"x\""
);

mk_fail_test!(
    duplicate_fun,
    "error::dupfun.egg",
    "multiple defined functions named \"f\""
);

// Egg-Eater: Array Tests
mk_fail_test!(
    array_empty_access,
    "arr::emptyaccess.egg",
    "out of bounds"
);

mk_fail_test!(
    array_outofbounds_low,
    "arr::low.egg",
    "out of bounds"
);

mk_fail_test!(
    array_outofbounds_high,
    "arr::high.egg",
    "out of bounds"
);

mk_fail_test!(
    array_outofbounds_nested,
    "arr::oob.egg",
    "out of bounds"
);

mk_fail_test!(
    array_indexed_nonarray,
    "arr::indexednonarr.egg",
    "indexed into non-array"
);

mk_fail_test!(
    array_index_notnum,
    "arr::indexnotnum.egg",
    "index not a number"
);

mk_test!(
    array_index_ok,
    "arr::indexok.egg",
    "[1, false, 3]"
);

mk_test!(
    array_empty,
    "arr::empty.egg",
    "[]"
);

mk_test!(
    array_singleton,
    "arr::single.egg",
    "[1]"
);

mk_test!(
    array_three,
    "arr::three.egg",
    "[1, true, 3]"
);

mk_test!(
    array_many,
    "arr::many.egg",
    "[1, 2, true, false, [1, 2], [[true], false]]"
);

mk_test!(
    array_unit,
    "arr::unit.egg",
    "[]"
);

mk_fail_test!(
    array_add1,
    "arr::add1.egg",
    "arithmetic expected a number"
);

mk_fail_test!(
    array_num_plus_tuple,
    "arr::numplustuple.egg",
    "arithmetic expected a number"
);

mk_test!(
    array_single_read,
    "arr::singleread.egg",
    "1"
);

mk_test!(
    array_many_reads,
    "arr::manyreads.egg",
    "0"
);

mk_test!(
    array_single_write,
    "arr::singlewrite.egg",
    "[10, 20, 900]"
);

mk_test!(
    array_nested_writes,
    "arr::nestedwrites.egg",
    "[[5, true, false, 6], [-20], [13, false]]"
);

mk_test!(
    array_eq1,
    "arr::eq1.egg",
    "true"
);

mk_test!(
    array_eq2,
    "arr::eq2.egg",
    "false"
);

mk_test!(
    array_eq3,
    "arr::eq3.egg",
    "false"
);

mk_test!(
    array_isarray1,
    "arr::isarray1.egg",
    "false"
);

mk_test!(
    array_isarray2,
    "arr::isarray2.egg",
    "true"
);

mk_test!(
    array_isarray3,
    "arr::isarray3.egg",
    "true"
);

mk_fail_test!(
    array_length_err,
    "arr::lenerr.egg",
    "length called with non-array"
);

mk_test!(
    array_length_empty,
    "arr::lenempty.egg",
    "0"
);

mk_test!(
    array_length_singleton,
    "arr::lensingle.egg",
    "1"
);

mk_test!(
    array_length_many,
    "arr::lenmany.egg",
    "3"
);

mk_test!(
    array_length_nested,
    "arr::lennested.egg",
    "4"
);

mk_test!(
    array_trues,
    "arr::trues.egg",
    "[true, [true, [true, []]]]"
);

mk_test!(
    array_nested_reads,
    "arr::nestedreads.egg",
    "6"
);

mk_test!(
    array_popcount,
    "arr::popcount.egg",
    "0\n0\n1\n2\n3"
);

mk_test!(
    array_cycle,
    "arr::cycle.egg",
    "[5, <loop>]"
);

mk_test!(
    array_sequence,
    "arr::seq.egg",
    "false\ntrue\n3"
);

mk_test!(
    array_print,
    "arr::print.egg",
    "[4, [true, 3]]\nfalse"
);


mk_test!(
    array_chained_writes,
    "arr::chainedwrites.egg",
    "[1, 10, 3]\n[[1, 10, 3], 0]"
);

mk_test!(
    array_sequenced_lets,
    "arr::seqlet.egg",
    "4\n30\nfalse\ntrue\n3"
);

mk_test!(
    array_write_read,
    "arr::writeread.egg",
    "false"
);

mk_test!(
    array_graph,
    "arr::graph.egg",
    "[[true, true], [true, true]]"
);

mk_test!(
    array_preserve,
    "arr::preserve.egg",
    "1"
);

mk_test!(
    array_true_print,
    "arr::trueprint.egg",
    "[1, [true, [true]]]"
);

// Interesting egg test case
mk_test!(
    interesting,
    "interesting.egg",
    "[0, 1, [4, <loop>, 5], [4, <loop>, 5]]"
);

// Error-Handling: Dynamic Errors
// mk_fail_test!(
//     wrong_arity_1,
//     "error::wrongarity1.egg",
//     "wrong number of arguments"
// );

// mk_fail_test!(
//     wrong_arity_2,
//     "error::wrongarity2.egg",
//     "wrong number of arguments"
// );

// mk_fail_test!(
//     nonfun_1,
//     "error::nonfun1.egg",
//     "called a non-function"
// );

// mk_fail_test!(
//     nonfun_2,
//     "error::nonfun2.egg",
//     "called a non-function"
// );

// mk_fail_test!(
//     nonfun_3,
//     "error::nonfun3.egg",
//     "called a non-function"
// );

// Egg Eater: Closure Tests
mk_test!(
    closure_print,
    "closure::print.egg",
    "[true, <closure>, 0]"
);

mk_test!(
    closure_isfun_false,
    "closure::isfunfalse.egg",
    "false"
);

// TODO: isfun true test

mk_test!(
    closure_apply,
    "closure::apply.egg",
    "5"
);

mk_fail_test!(
    closure_arity,
    "closure::arity.egg",
    "wrong number of arguments"
);

mk_fail_test!(
    closure_church_pairs,
    "closure::churchpairs.egg",
    "called a non-function"
);

mk_test!(
    closure_1,
    "closure::closure1.egg",
    "20"
);

mk_test!(
    closure_2,
    "closure::closure2.egg",
    "35"
);

mk_test!(
    closure_nested,
    "closure::nested.egg",
    "36"
);

mk_test!(
    closure_escaping,
    "closure::escaping.egg",
    "8"
);

mk_test!(
    closure_curried_subtraction,
    "closure::curriedsub.egg",
    "1"
);

mk_test!(
    closure_curried_addition,
    "closure::curriedadd.egg",
    "7"
);

mk_test!(
    closure_easy,
    "closure::easy.egg",
    "1"
);

mk_test!(
    closure_add1,
    "closure::add1.egg",
    "11"
);

mk_test!(
    closure_add,
    "closure::add.egg",
    "30"
);

mk_test!(
    closure_fold_left,
    "closure::foldleft.egg",
    "[6, 24, 120]"
);

mk_fail_test!(
    closure_duplicate_args,
    "closure::duparg.egg",
    "multiple arguments named \"f\""
);

mk_test!(
    closure_applications,
    "closure::aps.egg",
    "[2, 11]"
);

mk_test!(
    closure_compose,
    "closure::compose.egg",
    "102"
);

mk_test!(
    closure_big_compose,
    "closure::bigcompose.egg",
    "202"
);

mk_test!(
    closure_locals,
    "closure::locals.egg",
    "4"
);

mk_test!(
    closure_local2,
    "closure::local2.egg",
    "5"
);

mk_test!(
    closure_map,
    "closure::map.egg",
    "[2, 3]"
);

mk_fail_test!(
    closure_non_fun,
    "closure::nonfun.egg",
    "called a non-function"
);

mk_fail_test!(
    closure_even_odd,
    "closure::evenodd.egg",
    "true"
);

mk_test!(
    closure_two,
    "closure::two.egg",
    "2"
);

mk_test!(
    closure_lambda_application,
    "closure::lamap.egg",
    "2"
);

mk_test!(
    closure_big_applications,
    "closure::bigaps.egg",
    "1\n2\n3\n4\n5\n6\n7\n[1, [4, [[[9, 16]]], 25], [[36, 49]], []]"
);

mk_test!(
    closure_big_daddy,
    "closure::bigdaddy.egg",
    "120\n5\n6\n36\ntrue\nfalse\n42"
);

mk_test!(
    closure_ackermann,
    "closure::ackermann.egg",
    "7\ntrue"
);

mk_test!(
    closure_mutual,
    "closure::mutual.egg",
    "true"
);

mk_test!(
    closure_functional,
    "closure::functional.egg",
    "false\n<closure>\n0"
);

mk_test!(
    closure_three,
    "closure::three.egg",
    "3"
);

mk_test!(
    closure_tail,
    "closure::tail.egg",
    "true"
);

mk_test!(
    closure_spec,
    "closure::spec.egg",
    "3"
);

mk_test!(
    closure_notes,
    "closure::notes.egg",
    "12"
);

// IMPLEMENTATION
fn test_example_file(f: &str, expected_str: &str) -> std::io::Result<()> {
    use std::path::Path;
    let p_name = format!("examples/{}", f);
    let path = Path::new(&p_name);

    // Test the compiler
    let tmp_dir = tempfile::TempDir::new()?;
    let mut w = Vec::new();
    match runner::compile_and_run_file(&path, tmp_dir.path(), &mut w) {
        Ok(()) => {
            let stdout = std::str::from_utf8(&w).unwrap();
            assert_eq!(stdout.trim(), expected_str)
        }
        Err(e) => {
            assert!(false, "Expected {}, got an error: {}", expected_str, e)
        }
    }

    Ok(())
}

fn test_example_fail(f: &str, includes: &str) -> std::io::Result<()> {
    use std::path::Path;
    let p_name = format!("examples/{}", f);
    let path = Path::new(&p_name);

    // Test the compiler
    let tmp_dir = tempfile::TempDir::new()?;
    let mut w_run = Vec::new();
    match runner::compile_and_run_file(
        &Path::new(&format!("examples/{}", f)),
        tmp_dir.path(),
        &mut w_run,
    ) {
        Ok(()) => {
            let stdout = std::str::from_utf8(&w_run).unwrap();
            assert!(false, "Expected a failure but got: {}", stdout.trim())
        }
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains(includes),
                "Expected error message to include the string \"{}\" but got the error: {}",
                includes,
                msg
            )
        }
    }

    Ok(())
}

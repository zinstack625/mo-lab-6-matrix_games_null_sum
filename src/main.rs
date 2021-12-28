use log::trace;
use ndarray::*;
use simplex_method::Table;

fn get_simplex(matrix_strategy: ndarray::Array2<f64>) -> (Table, Table) {
    let min_win = {
        let mut min = None;
        for i in matrix_strategy.iter() {
            if min.is_none() || *min.as_ref().unwrap() > i {
                min = Some(i);
            }
        }
        min.unwrap().clone()
    };
    let mut constr_val = vec![min_win; matrix_strategy.ncols()];
    let mut func_coeff = vec![1f64; matrix_strategy.nrows()];
    let mut constr_coeff = matrix_strategy.reversed_axes();
    // Ax >= B
    for i in constr_val.iter_mut() {
        *i *= -1f64;
    }
    for i in constr_coeff.iter_mut() {
        *i *= -1f64;
    }
    let first_player = Table::new(
        constr_coeff.clone(),
        Array1::from(constr_val.clone()),
        Array1::from(func_coeff.clone()),
        true,
    );
    // Ay <= F
    for i in func_coeff.iter_mut() {
        *i *= -1f64;
    }
    let second_player = get_inverse_task(
        constr_coeff,
        Array1::from(constr_val),
        Array1::from(func_coeff),
    );
    (first_player, second_player)
}

fn get_inverse_task(
    mut constr_coeff: ndarray::Array2<f64>,
    constr_val: ndarray::Array1<f64>,
    mut func_coeff: ndarray::Array1<f64>,
) -> Table {
    // AX >= B <=> -AX <= -B
    for i in constr_coeff.iter_mut() {
        *i *= -1f64;
    }
    for i in func_coeff.iter_mut() {
        *i *= -1f64;
    }
    Table::new(constr_coeff.reversed_axes(), func_coeff, constr_val, true)
}

fn main() {
    env_logger::init();
    let matrix_strategy = array![
        [19f64, 6f64, 17f64, 9f64, 18f64],
        [16f64, 18f64, 13f64, 13f64, 12f64],
        [11f64, 1f64, 5f64, 3f64, 12f64],
        [4f64, 5f64, 15f64, 19f64, 4f64]];
    let (mut first_player, mut second_player) = get_simplex(matrix_strategy);
    trace!("First player:");
    let err_forward = first_player.optimise();
    trace!("Second player:");
    let err_inverse = second_player.optimise();
    if err_forward.is_ok() && err_inverse.is_ok() {
        println!("First player:\n{}", first_player);
        println!("Second player:\n{}", second_player);
    } else if err_forward.is_err() && err_inverse.is_err() {
        println!(
            "First error: {:?}\tSecond error: {:?}",
            err_forward.unwrap_err(),
            err_inverse.unwrap_err()
        );
    } else {
        panic!("Stuff went horribly wrong");
    }
}

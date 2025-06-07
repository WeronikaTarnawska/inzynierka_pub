/* ==================================================================================================
 *                           This file is part of the bachelor thesis project
 *                  Implementation and Analysis of Selected Noise Reduction Methods
 *                                Weronika Tarnawska (Index No. 331171)
 *                                  Supervisor:  dr hab. Paweł Woźny
 *                                  University of Wrocław, June 2025
 * ================================================================================================== */
pub mod wav;
pub mod utils;
pub mod signal;
pub mod fft;
pub mod lms;
pub mod wiener;
pub mod ss;

#[cfg(test)]
const _EPSILON: f64 = 1e-12;

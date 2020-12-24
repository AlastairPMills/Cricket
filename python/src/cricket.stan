data {
  int<lower=1> N; // number of balls
  int<lower=1> B; // number of batsman
  int<lower=1> K; // number of bowlers
  int<lower=1,upper=B> batsman[N];
  int<lower=1,upper=K> bowler[N];
  int<lower=0,upper=1> result[N];
  vector<lower=0>[N] weight;
  vector[B] starting_guess_bat;
  vector[K] starting_guess_bowl;
  real<lower=0> sigma_bat;
  real<lower=0> sigma_bowl;
}
parameters {
  // parameters
  vector[B] bat_ability;
  vector[K] bowl_ability;
  real base_rate;
}
model {
  // priors
  bat_ability ~ normal(starting_guess_bat, sigma_bat);
  bowl_ability ~ normal(starting_guess_bowl, sigma_bowl);
  // likelihood
  for (n in 1:N){
    target += weight[n] * bernoulli_logit_lpmf(result[n] | base_rate
                                                           + bat_ability[batsman[n]] - bowl_ability[bowler[n]]);
  }
}

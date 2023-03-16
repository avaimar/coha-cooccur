library(tidyverse)
library(reshape2)
library(data.table)
library(ggplot2)
library(sandwich)
library(lmtest)

# Parameters
# * Matches the computation of Garg et al. (it fails to check if otherization words
# have non-zero vectors) 
match_garg <- 'False' 
# * Cosine vs RND scores
cosine_sim <- 'True'
# Note: if RND, then Metric = Norm (Distance). If cosine, them Metric = CS (Similarity)
# so outcomes will have opposite signs depending on the distance measure. 

vectors <- 'SPPMI'

# 1. Estimate models and impute bias scores --------------
# Define regression file
regression_file <- paste0(
  'results/', vectors, '/regressions/regression_df_MG', match_garg, '_CS', cosine_sim, '.csv')

# Load data
regression_df <- read.csv(regression_file)
regression_df <- regression_df %>%
  select(-Experiment) %>%
  select(-X.Deviation.Otherization..quartile.) %>% select(-X.Deviation.Target..quartile.)

# Decades
print('Decades')
print(sort(unique(regression_df$decade)))
first_decade <- min(regression_df$decade)

# * Single surnames
surname_df <- regression_df %>% filter(other_word != '--Overall--')

# * Aggregate bias scores (these use the mean surname vector)
aggregate_df <- regression_df %>% filter(other_word == '--Overall--')
aggregate_df <- dplyr::select(aggregate_df, wl, decade, bias_score)

df <- data.table()
for (wlist in c('{} San Bruno All', 'PNAS {} Target Words')) {
  wl_df <- filter(surname_df, wl == wlist) %>% select(-wl) %>% select(-bias_score)
  
  # Drop missing values (these stem from the match_garg option)
  wl_df <- wl_df %>%
    filter(!is.na(X.Deviation.Otherization))
  
  # Unadjusted **
  model <- lm(Metric ~ Group, data=wl_df)
  
  model_r <- coeftest(
    model, vcov = vcovCL, cluster = ~other_word)
  print('[INFO] Unadjusted model')
  print(model_r)
  
  # Adjusted ** 
  model <- lm(
    Metric ~ Group + X.Deviation.Otherization * X.Deviation.Target, 
    data=wl_df)
  
  model_r <- coeftest(
    model, vcov = vcovCL, cluster = ~other_word)
  print('[INFO] Adjusted model')
  print(model_r)
  
  # Imputation model
  # The interaction between the Otherization deviation and the Target deviation seems
  # to be the driving factor. 
  model <- lm(
    Metric ~ Group + X.Deviation.Otherization * X.Deviation.Target + factor(decade) * Group, 
    data=wl_df)
  
  # Verify base coefficient
  stopifnot(!(paste0('factor(decade)', first_decade) %in% names(coef(model))))
  
  # Compute bias for each decade assuming X.Deviation.Target is the same for
  # both groups
  imputation_df <- data.table()
  for (decade in unique(wl_df$decade)) {
    interaction_term <- coef(model)[paste0('GroupWhite:factor(decade)', decade)]
    if (decade == first_decade) interaction_term <- 0
    bias_score_decade <- coef(model)['GroupWhite'] + interaction_term
    if (cosine_sim == 'True') bias_score_decade <- -1 * bias_score_decade
    
    imputation_df <- rbind(
      imputation_df, data.table(
        wl=wlist,
        decade=decade, 
        bias_type='imputed',
        bias_score=bias_score_decade))
  }
  
  #true_bias_scores <- dplyr::filter(aggregate_df, wl==wlist) %>%
  #  dplyr::mutate(bias_type='true') %>%
  #  dplyr::select(wl, decade, bias_type, bias_score)
  df <- df %>% rbind(imputation_df)
  
}

# Add original bias scores (the ones computed on the aggregation of disag. scores)
group_components <- surname_df %>%
  dplyr::group_by(wl, decade, Group, other_word) %>%
  dplyr::summarize(Metric = mean(Metric)) %>%
  dplyr::ungroup()

group_components <- 
  data.table::dcast(
    data.table(group_components), 
    wl + decade + other_word ~ Group, 
    value.var=c('Metric'))

group_components <- group_components %>%
  dplyr::mutate(bias_score = Asian - White)

# Average value across otherization words
group_components <- group_components %>%
  group_by(wl, decade) %>%
  summarize(bias_score = mean(bias_score)) %>%
  ungroup()
group_components <- group_components %>% dplyr::mutate(bias_type='true')
df <- rbind(df, dplyr::select(group_components, wl, decade, bias_type, bias_score))

# Add aggregate bias scores (the ones computed using the mean vector)
df <- rbind(
  df, 
  aggregate_df %>% dplyr::mutate(bias_type='mean_vector'))


# Plot
df <- df %>% dplyr::mutate(
  wl = case_when(wl == '{} San Bruno All' ~ 'San Bruno', 
                 TRUE ~ 'PNAS')) %>%
  dplyr::mutate(bias_type = factor(bias_type, levels=c('true', 'imputed', 'mean_vector')))


ggplot(data=dplyr::filter(df, bias_type %in% c('true', 'imputed')), aes(x=decade, y=bias_score)) + 
  geom_line(aes(linetype=bias_type, color=wl)) + 
  facet_grid(cols=vars(wl)) + 
  geom_hline(yintercept=0, color='black') + 
  theme_light() +
  labs(color='Word List', x='Decade', y='Bias score') + 
  theme(legend.position="bottom") + 
  scale_linetype_discrete(
    'Bias score', 
    labels=c('Original (Individual surnames)', 'Imputed'))
ggsave(paste0('results/', vectors, '/regressions/bscores_MG', match_garg, '_CS', cosine_sim, '.png'),  
       width = 7, height = 4, dpi = 200, units = "in",)

ggplot(data=dplyr::filter(df, bias_type %in% c('true', 'mean_vector')), aes(x=decade, y=bias_score)) + 
  geom_line(aes(color=bias_type)) + 
  facet_grid(cols=vars(wl)) + 
  geom_hline(yintercept=0, color='black') + 
  theme_light() +
  labs(color='Word List', x='Decade', y='Bias score') + 
  theme(legend.position="bottom") + 
  scale_color_discrete(
    'Bias score', 
    labels=c('Original (Individual surnames)', 'Original (Mean Vector)'))
ggsave(paste0('results/', vectors, '/regressions/comp_bscores_MG', match_garg, '_CS', cosine_sim, '.png'),  
       width = 7, height = 4, dpi = 200, units = "in",)


# 2. Check aggregate vs singular results ---------------------
# Average value across surnames
group_components <- surname_df %>%
  dplyr::group_by(wl, decade, Group, other_word) %>%
  dplyr::summarize(Metric = mean(Metric)) %>%
  dplyr::ungroup()

group_components <- 
  data.table::dcast(
    data.table(group_components), 
    wl + decade + other_word ~ Group, 
    value.var=c('Metric'))

group_components <- group_components %>%
  dplyr::mutate(bias_score = Asian - White)

# Average value across otherization words
group_components <- group_components %>%
  group_by(wl, decade) %>%
  summarize(bias_score = mean(bias_score)) %>%
  ungroup()

joined_df <- aggregate_df %>%
  left_join(group_components, by=c('wl', 'decade'))

# 3. Check bias score trendlines -----------------------
regression_df_RND <- read.csv(
  paste0('results/HistWords/regressions/regression_df_MG', match_garg, '_CSFalse', '.csv'))
aggregate_df_RND <- regression_df_RND %>% filter(other_word == '--Overall--')
aggregate_df_RND <- dplyr::select(aggregate_df_RND, wl, decade, bias_score)
aggregate_df_RND <- dplyr::mutate(aggregate_df_RND, score='RND')

regression_df_CS <- read.csv(
  paste0('results/HistWords/regressions/regression_df_MG', match_garg, '_CSTrue', '.csv'))
aggregate_df_CS <- regression_df_CS %>% filter(other_word == '--Overall--')
aggregate_df_CS <- dplyr::select(aggregate_df_CS, wl, decade, bias_score)
aggregate_df_CS <- dplyr::mutate(aggregate_df_CS, score='CS')

aggregate_df <- aggregate_df_RND %>% rbind(
  aggregate_df_CS)

ggplot(aggregate_df, aes(x=decade, y=bias_score, group=wl)) + 
  facet_grid(cols=vars(score)) + geom_line(aes(color=wl)) + 
  theme_light() + labs(color='Word List', x='Decade', y='Bias score') 

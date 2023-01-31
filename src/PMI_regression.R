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

# 1. Estimate models and impute bias scores --------------
# Load data
regression_df <- read.csv(
  paste0('results/PMI/regression_df_MG', match_garg, '_CS', cosine_sim, '.csv'))
regression_df <- regression_df %>%
  select(-Experiment)

# * Single surnames
surname_df <- regression_df %>% filter(other_word != '--Overall--')

# * Aggregate bias scores
aggregate_df <- regression_df %>% filter(other_word == '--Overall--')
aggregate_df <- dplyr::select(aggregate_df, wl, decade, bias_score)

df <- data.table()
for (wlist in c('{} San Bruno All', 'PNAS {} Target Words')) {
  wl_df <- filter(surname_df, wl == wlist) %>% select(-wl) %>% select(-bias_score)
  
  # Drop missing values (these stem from the match_garg option)
  wl_df <- wl_df %>%
    filter(!is.na(X.Deviation.Otherization))
  
  # Unadjusted **
  model <- lm(
    Metric ~ Group, 
    data=filter(wl_df, X.Deviation.Target..quartile. != 4))
  
  model_r <- coeftest(
    model, vcov = vcovCL, cluster = ~other_word)
  print('[INFO] Unadjusted model')
  print(model_r)
  
  # Adjusted ** 
  model <- lm(
    Metric ~ Group + X.Deviation.Otherization * X.Deviation.Target, 
    data=filter(wl_df, X.Deviation.Target..quartile. != 4))
  
  model_r <- coeftest(
    model, vcov = vcovCL, cluster = ~other_word)
  print('[INFO] Adjusted model')
  print(model_r)
  
  # Imputation model
  # The interaction between the Otherization deviation and the Target deviation seems
  # to be the driving factor. 
  model <- lm(
    Metric ~ Group + X.Deviation.Otherization * X.Deviation.Target + factor(decade) * Group, 
    data=filter(wl_df, X.Deviation.Target..quartile. != 4))
  
  # Compute bias for each decade assuming X.Deviation.Target is the same for
  # both groups
  imputation_df <- data.table()
  for (decade in unique(wl_df$decade)) {
    interaction_term <- coef(model)[paste0('GroupWhite:factor(decade)', decade)]
    if (decade == 1920) interaction_term <- 0
    
    imputation_df <- rbind(
      imputation_df, data.table(
        wl=wlist,
        decade=decade, 
        bias_type='imputed',
        bias_score=coef(model)['GroupWhite'] + interaction_term))
  }
  
  # Add original bias scores
  true_bias_scores <- dplyr::filter(aggregate_df, wl==wlist) %>%
    dplyr::mutate(bias_type='true') %>%
    dplyr::select(wl, decade, bias_type, bias_score)
  imputation_df <- imputation_df %>% rbind(true_bias_scores)
  
  df <- rbind(df, imputation_df)
}

# Plot
df <- df %>% dplyr::mutate(
  wl = case_when(wl == '{} San Bruno All' ~ 'San Bruno', 
                 TRUE ~ 'PNAS')) %>%
  dplyr::mutate(bias_type = factor(bias_type, levels=c('true', 'imputed')))


ggplot(data=df, aes(x=decade, y=bias_score)) + 
  geom_line(aes(linetype=bias_type, color=wl)) + 
  facet_grid(cols=vars(wl)) + 
  geom_hline(yintercept=0, color='black') + 
  theme_light() +
  labs(color='Word List', x='Decade', y='Bias score') + 
  theme(legend.position="bottom") + 
  scale_linetype_discrete('Bias score', labels=c('Original', 'Imputed'))
ggsave(paste0('results/PMI/regressions/bscores_MG', match_garg, '_CS', cosine_sim, '.png'),  
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
  paste0('results/PMI/regression_df_MG', match_garg, '_CSFalse', '.csv'))
aggregate_df_RND <- regression_df_RND %>% filter(other_word == '--Overall--')
aggregate_df_RND <- dplyr::select(aggregate_df_RND, wl, decade, bias_score)
aggregate_df_RND <- dplyr::mutate(aggregate_df_RND, score='RND')

regression_df_CS <- read.csv(
  paste0('results/PMI/regression_df_MG', match_garg, '_CSTrue', '.csv'))
aggregate_df_CS <- regression_df_CS %>% filter(other_word == '--Overall--')
aggregate_df_CS <- dplyr::select(aggregate_df_CS, wl, decade, bias_score)
aggregate_df_CS <- dplyr::mutate(aggregate_df_CS, score='CS')

aggregate_df <- aggregate_df_RND %>% rbind(
  aggregate_df_CS)

ggplot(aggregate_df, aes(x=decade, y=bias_score, group=wl)) + 
  facet_grid(cols=vars(score)) + geom_line(aes(color=wl)) + 
  theme_light() + labs(color='Word List', x='Decade', y='Bias score') 

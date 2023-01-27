library(tidyverse)
library(reshape2)
library(data.table)
library(ggplot2)

match_garg <- 'False'
regression_df <- read.csv(paste0('regression_df_MG', match_garg, '.csv'))
regression_df <- regression_df %>%
  select(-Experiment)

for (wlist in c('{} San Bruno All', 'PNAS {} Target Words')) {
  wl_df <- filter(regression_df, wl == wlist) %>% select(-wl) 
  
  # Drop missing values (these stem from the match_garg option)
  wl_df <- wl_df %>%
    filter(!is.na(X.Deviation.Otherization))
  
  #model <- lm(Norm ~ Group + X.Deviation.Otherization + X.Deviation.Target, 
  #            data=wl_df)
  
  # The interaction between the Otherization deviation and the Target deviation seems
  # to be the driving factor. 
  model <- lm(
    Norm ~ Group + X.Deviation.Otherization * X.Deviation.Target + factor(decade), 
    data=filter(wl_df, X.Deviation.Target..quartile. != 4))
  
  #model <- lm(Norm ~ Group + X.Deviation.Otherization + X.Deviation.Target + factor(decade) , data=wl_df)
  #model <- lm(Norm ~ Group + factor(X.Deviation.Otherization..quartile.) + factor(X.Deviation.Target..quartile.) + factor(decade), data=wl_df)
  #model <- lm(Norm ~ Group * factor(X.Deviation.Target..quartile.) + factor(X.Deviation.Otherization..quartile.) + factor(decade), data=wl_df)
  
  # What percentage of the bias score is explained by the EError?
  # Estimate bias scores 
  wl_df_dcast <- 
    data.table::dcast(
      data.table(wl_df), 
      decade + other_word + bias_score + X.Deviation.Otherization + X.Deviation.Otherization..quartile. ~ Group, 
      value.var=c('Norm', 'X.Deviation.Target', 'X.Deviation.Target..quartile.'))
  
  # Get adjusted Asian norms
  wl_asian <- wl_df_dcast %>%
    dplyr::mutate(X.Deviation.Target = X.Deviation.Target_White) %>%
    select(decade, other_word, X.Deviation.Target, X.Deviation.Otherization) %>%
    dplyr::mutate(Group = 'Asian')
  
  wl_asian <- wl_asian %>%
    dplyr::mutate(Norm_Asian_pred = predict(model, wl_asian)) %>%
    dplyr::mutate(X.Deviation.Target_White = X.Deviation.Target)
  
  wl_df_dcast <- wl_df_dcast %>%
    left_join(select(wl_asian, decade, other_word, X.Deviation.Target_White, Norm_Asian_pred), 
              by=c('decade', 'other_word', 'X.Deviation.Target_White'))
  
  # Compute new bias scores
  wl_df_dcast <- wl_df_dcast %>%
    dplyr::mutate(bias_score_adj = Norm_White - Norm_Asian_pred)
  
  # Aggregate across otherization words
  wl_df_bscores <- wl_df_dcast %>%
    dplyr::group_by(decade, X.Deviation.Target..quartile._White) %>%
    dplyr::summarise(bias_score = mean(bias_score), bias_score_adj = mean(bias_score_adj)) %>%
    dplyr::ungroup()
  
  # Melt
  wl_df_bscores_melt <- data.table::melt(
    data.table(wl_df_bscores), id.vars=c('decade', 'X.Deviation.Target..quartile._White'), 
    measure.vars=c('bias_score', 'bias_score_adj'), variable.name='Bias_score')
  
  ggplot(data=wl_df_bscores_melt, aes(x=decade, group=Bias_score)) + 
    geom_line(aes(y=value, linetype=Bias_score, color=factor(X.Deviation.Target..quartile._White))) + 
    facet_grid(cols=vars(factor(X.Deviation.Target..quartile._White))) + 
    geom_hline(yintercept=0, color='black') + 
    theme_light() +
    labs(color='Target %Dev Quartile', x='Decade', y='Bias score') + 
    theme(legend.position="bottom") + 
    scale_color_discrete(labels=c('0', '1', '2', '3', 'All')) +
    scale_linetype_discrete('Bias score', labels=c('Original', 'Adjusted'))
  ggsave(paste0('regressions/bscores_MG', match_garg, '_wl', wlist, '.png'),  
         width = 10, height = 4, dpi = 150, units = "in",)
}

summary(model)


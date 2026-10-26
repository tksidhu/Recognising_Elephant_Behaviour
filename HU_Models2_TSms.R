############################################################
# Models for the Human Understanding of Elephant Behaviour Study
# Owner: Tapinder Sidhu
# Created: 13/05/2026
# Updated: 26/08/2026
#
# Main research questions:
# Can participants identify valence and behaviour significanly above chance?
# Does participant group experience improve accuracy?
# Do videos improve accuracy compared with still images?
# Are negative Valence or behaviours recognised more accurately?
############################################################


rm(list = ls())

library(tidyverse)
library(lme4)
library(performance)
library(performance)
library(emmeans)


# Read in CSV file ####


fulldata <- read.csv(
  "Data/HU_FullTaskDataset.csv",
  header = TRUE,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA")
)

dir.create("Data/Model_outputs", showWarnings = FALSE, recursive = TRUE)
dir.create("Figures", showWarnings = FALSE, recursive = TRUE)


# Check required variables ####


required_vars <- c(
  "Participant_ID",
  "Event_ID",
  "Group",
  "StimulusType",
  "StimulusValence",
  "BehaviourCategory",
  "SpecificBehaviour",
  "Trial_No",
  "CorrectVal",
  "CorrectBeh"
)

missing_vars <- setdiff(required_vars, names(fulldata))

if (length(missing_vars) > 0) {
  stop("Missing variables: ", paste(missing_vars, collapse = ", "))
}


# Prepare variables ####


HU_model <- fulldata %>%
  mutate(
    Participant_ID = factor(Participant_ID),
    Event_ID = factor(Event_ID),
    Trial_No = factor(Trial_No),
    
    ParticipantGroup = factor(
      Group,
      levels = c("Novice", "Live/work alongside", "Work directly with")
    ),
    
    StimulusType = factor(
      StimulusType,
      levels = c("Image", "Video")
    ),
    
    StimulusValence = factor(
      StimulusValence,
      levels = c("Neutral", "Negative")
    ),
    
    BehaviourCategory = factor(
      BehaviourCategory,
      levels = c(
        "Continue",
        "Attention",
        "Retreat",
        "Aggression"
      )
    ),
    
    SpecificBehaviour = factor(SpecificBehaviour),
    
    CorrectVal = as.numeric(CorrectVal),
    CorrectBeh = as.numeric(CorrectBeh),
    
    BehaviourChanceOffset = qlogis(0.25) # compute logit odds from proabability (no need to do for valence as logit of 0.5 / 50% chance = 0 already)
  ) %>%
  filter(
    !is.na(Participant_ID),
    !is.na(Event_ID),
    !is.na(Trial_No),
    !is.na(ParticipantGroup),
    !is.na(StimulusType),
    !is.na(StimulusValence),
    !is.na(BehaviourCategory),
    !is.na(CorrectVal),
    !is.na(CorrectBeh)
  )


# Basic checks ####


table(HU_model$ParticipantGroup)
table(HU_model$StimulusType)
table(HU_model$StimulusValence)
table(HU_model$BehaviourCategory)
table(HU_model$Trial_No)

xtabs(~ ParticipantGroup + StimulusType, data = HU_model)
xtabs(~ ParticipantGroup + StimulusValence, data = HU_model)
xtabs(~ StimulusValence + BehaviourCategory, data = HU_model)
xtabs(~ ParticipantGroup + StimulusType + StimulusValence, data = HU_model)
xtabs(~ ParticipantGroup + StimulusType + StimulusValence + BehaviourCategory, data = HU_model)

xtabs(~ CorrectVal, data = HU_model)
xtabs(~ CorrectBeh, data = HU_model)
xtabs(~ StimulusValence + CorrectVal, data = HU_model)
xtabs(~ StimulusValence + BehaviourCategory + CorrectBeh, data = HU_model)

colSums(is.na(HU_model))


# ---------------------------------------------------------
# Helper functions ####
# ---------------------------------------------------------


fit_glmm <- function(formula, data = HU_model) {
  glmer(
    formula,
    data = data,
    family = binomial(link = "logit"),
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 1e5)
    )
  )
}

extract_lrt <- function(lrt_object, effect_name) {
  tibble(
    Effect = effect_name,
    Chisq = lrt_object$Chisq[2],
    Df = lrt_object$`Chi Df`[2],
    p_value = lrt_object$`Pr(>Chisq)`[2]
  )
}

# This helper keeps only the four real combinations in the design.
filter_observed_valence_behaviour <- function(df) {
  df %>%
    filter(
      (StimulusValence == "Neutral" & BehaviourCategory %in% c("Continue", "Attention")) |
        (StimulusValence == "Negative" & BehaviourCategory %in% c("Retreat", "Aggression"))
    )
}


# Set contrasts and reference levels ####


HU_model$ParticipantGroup <- relevel(HU_model$ParticipantGroup, ref = "Novice")
HU_model$StimulusType <- relevel(HU_model$StimulusType, ref = "Image")
HU_model$StimulusValence <- relevel(HU_model$StimulusValence, ref = "Neutral")
HU_model$BehaviourCategory <- relevel(HU_model$BehaviourCategory, ref = "Continue")

contrasts(HU_model$ParticipantGroup) <- contr.treatment(
  nlevels(HU_model$ParticipantGroup),
  base = which(levels(HU_model$ParticipantGroup) == "Novice")
)

# Novice     0 0
# Living     1 0
# Experience 0 1
# Levels: Novice Living Experience

contrasts(HU_model$StimulusType) <- contr.treatment(
  nlevels(HU_model$StimulusType),
  base = which(levels(HU_model$StimulusType) == "Image")
)

contrasts(HU_model$StimulusValence) <- contr.treatment(
  nlevels(HU_model$StimulusValence),
  base = which(levels(HU_model$StimulusValence) == "Neutral")
)

contrasts(HU_model$BehaviourCategory) <- contr.treatment(
  nlevels(HU_model$BehaviourCategory),
  base = which(levels(HU_model$BehaviourCategory) == "Continue")
)

# Continue   0 0 0
# Attention 1 0 0
# Retreat    0 1 0
# Aggression 0 0 1
# Levels: Continue Attention Retreat Aggression


# Above-chance models ####


Valence_chance_model <- fit_glmm(
  CorrectVal ~ 1 +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

summary(Valence_chance_model)

Valence_chance_summary <- tibble(
  Outcome = "Valence",
  Chance = 0.50,
  Chance_logit = qlogis(0.50),
  Estimate_logit = fixef(Valence_chance_model)[1],
  Estimate_probability = plogis(fixef(Valence_chance_model)[1])
)

write_csv(Valence_chance_summary, "Data/Valence_above_chance_summary.csv")

behaviour_chance_model <- glmer(
  CorrectBeh ~ 1 +
    offset(BehaviourChanceOffset) +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No),
  data = HU_model,
  family = binomial(link = "logit"),
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 1e5)
  )
)

summary(behaviour_chance_model)

behaviour_chance_summary <- tibble(
  Outcome = "Behaviour",
  Chance = 0.25,
  Chance_logit = qlogis(0.25),
  Estimate_above_chance_logit = fixef(behaviour_chance_model)[1],
  Estimate_probability = plogis(
    qlogis(0.25) + fixef(behaviour_chance_model)[1]
  )
)

write_csv(behaviour_chance_summary, "Data/behaviour_above_chance_summary.csv")

saveRDS(Valence_chance_model, "Data/Model_outputs/Valence_above_chance_model.rds")
saveRDS(behaviour_chance_model, "Data/Model_outputs/behaviour_above_chance_model.rds")

capture.output(
  summary(Valence_chance_model),
  file = "Data/Model_outputs/Valence_above_chance_model_summary.txt"
)

capture.output(
  summary(behaviour_chance_model),
  file = "Data/Model_outputs/behaviour_above_chance_model_summary.txt"
)


# Get means and CIs
valchance_emm <- emmeans(Valence_chance_model,   ~ 1, type = "response")
valchance_emm
behchance_emm <- emmeans(behaviour_chance_model, ~ 1, type = "response")
behchance_emm

# raw overall success rate
mean(HU_model$CorrectVal)
mean(HU_model$CorrectBeh)

# per-participant accuracy, then mean and SD across participants
HU_model %>%
  group_by(Participant_ID) %>%
  summarise(acc = mean(CorrectVal), .groups = "drop") %>%
  summarise(mean = mean(acc), sd = sd(acc), n = n())

# per-participant accuracy, then mean and SD across participants
HU_model %>%
  group_by(Participant_ID) %>%
  summarise(acc = mean(CorrectBeh), .groups = "drop") %>%
  summarise(mean = mean(acc), sd = sd(acc), n = n())



# Valence accuracy model ####


Valence_model <- fit_glmm(
  CorrectVal ~ ParticipantGroup*StimulusType +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

summary(Valence_model)

Valence_modelfinal <- fit_glmm(
  CorrectVal ~ ParticipantGroup +
    StimulusType +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

summary(Valence_modelfinal)



# Behaviour accuracy model ####


behaviour_model <- fit_glmm(
  CorrectBeh ~ ParticipantGroup * StimulusType +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

summary(behaviour_model)

behaviour_modelfinal <- fit_glmm(
  CorrectBeh ~ ParticipantGroup +
    StimulusType +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

summary(behaviour_modelfinal)



# Model diagnostics ####


# collinearity 
check_collinearity(Valence_modelfinal)
check_collinearity(behaviour_modelfinal)

# random-effects structure and convergence
check_singularity(Valence_modelfinal);   check_convergence(Valence_modelfinal)
check_singularity(behaviour_modelfinal); check_convergence(behaviour_modelfinal)


# Save model objects and summaries ####


saveRDS(Valence_model, "Data/Model_outputs/Valence_model_interaction_nested.rds")
saveRDS(Valence_modelfinal, "Data/Model_outputs/Valence_model_final_nested.rds")
saveRDS(behaviour_model, "Data/Model_outputs/behaviour_model_interaction_nested.rds")
saveRDS(behaviour_modelfinal, "Data/Model_outputs/behaviour_model_final_nested.rds")

capture.output(
  summary(Valence_model),
  file = "Data/Model_outputs/Valence_model_interaction_nested_summary.txt"
)

capture.output(
  summary(Valence_modelfinal),
  file = "Data/Model_outputs/Valence_model_final_nested_summary.txt"
)

capture.output(
  summary(behaviour_model),
  file = "Data/Model_outputs/behaviour_model_interaction_nested_summary.txt"
)

capture.output(
  summary(behaviour_modelfinal),
  file = "Data/Model_outputs/behaviour_model_final_nested_summary.txt"
)




# Likelihood-ratio tests for Full vs null model comparisons


# Valence null model
Valence_model_null <- fit_glmm(
  CorrectVal ~ 1 +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

# Compare null vs full valence model
Valence_full_vs_null <- anova(
  Valence_model_null,
  Valence_modelfinal,
  test = "Chisq"
)

Valence_full_vs_null


# Behaviour null model
behaviour_model_null <- fit_glmm(
  CorrectBeh ~ 1 +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

# Compare null vs full behaviour model
behaviour_full_vs_null <- anova(
  behaviour_model_null,
  behaviour_modelfinal,
  test = "Chisq"
)

behaviour_full_vs_null



# Save results


full_vs_null_results <- bind_rows(
  as.data.frame(Valence_full_vs_null) %>%
    mutate(Model = "Valence"),
  as.data.frame(behaviour_full_vs_null) %>%
    mutate(Model = "Behaviour")
)

write_csv(
  full_vs_null_results,
  "Data/full_vs_null_model_comparisons.csv"
)



# Likelihood-ratio tests for each predictor ####


Valence_model_full <- Valence_modelfinal

Valence_model_no_group <- fit_glmm(
  CorrectVal ~ StimulusType +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

Valence_model_no_type <- fit_glmm(
  CorrectVal ~ ParticipantGroup +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

Valence_model_no_Valencebeh <- fit_glmm(
  CorrectVal ~ ParticipantGroup +
    StimulusType +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

Valence_LRT_table <- bind_rows(
  extract_lrt(
    anova(Valence_model_no_group, Valence_model_full, test = "Chisq"),
    "ParticipantGroup"
  ),
  extract_lrt(
    anova(Valence_model_no_type, Valence_model_full, test = "Chisq"),
    "StimulusType"
  ),
  extract_lrt(
    anova(Valence_model_no_Valencebeh, Valence_model_full, test = "Chisq"),
    "StimulusValence"
  )
)

write_csv(
  Valence_LRT_table,
  "Data/valence_model_likelihood_ratio_tests_nested.csv"
)

behaviour_model_full <- behaviour_modelfinal

behaviour_model_no_group <- fit_glmm(
  CorrectBeh ~ StimulusType +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

behaviour_model_no_type <- fit_glmm(
  CorrectBeh ~ ParticipantGroup +
    StimulusValence/BehaviourCategory +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

behaviour_model_no_Valencebeh <- fit_glmm(
  CorrectBeh ~ ParticipantGroup +
    StimulusType +
    (1 | Participant_ID) +
    (1 | Event_ID) +
    (1 | Trial_No)
)

behaviour_LRT_table <- bind_rows(
  extract_lrt(
    anova(behaviour_model_no_group, behaviour_model_full, test = "Chisq"),
    "ParticipantGroup"
  ),
  extract_lrt(
    anova(behaviour_model_no_type, behaviour_model_full, test = "Chisq"),
    "StimulusType"
  ),
  extract_lrt(
    anova(behaviour_model_no_Valencebeh, behaviour_model_full, test = "Chisq"),
    "StimulusValence/BehaviourCategory"
  )
)

write_csv(
  behaviour_LRT_table,
  "Data/behaviour_model_likelihood_ratio_tests_nested.csv"
)



# Estimated marginal means (nested reference grids) ####

# BehaviourCategory is nested within StimulusValence: only four
# combinations exist in the design (Neutral-Continue, Neutral-Attention, Negative-Retreat, Negative-Aggression). 
# This step restricts the reference grid to those four cells, so all marginal means are
# averaged on the logit scale with correctly propagated standard errors.

dir.create("DataEMMs", showWarnings = FALSE, recursive = TRUE)

val_grid <- ref_grid(
  Valence_model_full,
  nesting = list(BehaviourCategory = "StimulusValence")
)

beh_grid <- ref_grid(
  behaviour_model_full,
  nesting = list(BehaviourCategory = "StimulusValence")
)

# Sanity check: each should print 4 rows, not 8
emmeans(val_grid, ~ StimulusValence:BehaviourCategory)
emmeans(beh_grid, ~ StimulusValence:BehaviourCategory)

# ---------------------------------------------------------
# Helper: standard set of EMMs for one model ####
# ---------------------------------------------------------

emm_outputs <- function(grid) {
  cells <- emmeans(grid, ~ StimulusValence:BehaviourCategory)
  
  list(
    cells        = summary(cells, type = "response"),
    group        = emmeans(grid, pairwise ~ ParticipantGroup,
                           type = "response", adjust = "tukey"),
    type         = emmeans(grid, pairwise ~ StimulusType,
                           type = "response", adjust = "tukey"),
    valence      = emmeans(grid, pairwise ~ StimulusValence,
                           type = "response"),
    within_pairs = pairs(emmeans(grid, ~ BehaviourCategory | StimulusValence),
                         adjust = "tukey")
  )
}

val_emms <- emm_outputs(val_grid)
beh_emms <- emm_outputs(beh_grid)



# Pairwise comparisons ####

# All six pairwise comparisons among the four observed cells
# Contrasts computed on the logit scale and back-transformed

cell_pairs <- function(grid) {
  cells <- regrid( # regrid() produces an ordinary (un-nested) grid containing only the four real cells, for comparisons across valence levels
    emmeans(grid, ~ StimulusValence:BehaviourCategory),
    transform = "none"
  )
  contrast(cells, "pairwise", adjust = "mvt")
}

Valence_cell_pairs   <- cell_pairs(val_grid)
behaviour_cell_pairs <- cell_pairs(beh_grid)

nrow(as.data.frame(Valence_cell_pairs))    # expect 6
nrow(as.data.frame(behaviour_cell_pairs))  # expect 6


# Write outputs


save_emm <- function(obj, path) {
  if (is.null(obj)) {
    warning("NULL object, not written: ", path)
    return(invisible(NULL))
  }
  df <- as.data.frame(obj)
  if (nrow(df) == 0) {
    warning("Zero rows, not written: ", path)
    return(invisible(NULL))
  }
  write_csv(df, path)
  message("Wrote ", path, " (", nrow(df), " rows)")
}

# Valence model
save_emm(val_emms$cells,
         "DataEMMs/Valence_emmeans_observed_valence_by_behaviour_category.csv")
save_emm(val_emms$group$emmeans,
         "DataEMMs/Valence_emmeans_participant_group.csv")
save_emm(val_emms$group$contrasts,
         "DataEMMs/Valence_pairwise_participant_group.csv")
save_emm(val_emms$type$emmeans,
         "DataEMMs/Valence_emmeans_stimulitype.csv")
save_emm(val_emms$type$contrasts,
         "DataEMMs/Valence_pairwise_stimulitype.csv")
save_emm(val_emms$valence$emmeans,
         "DataEMMs/Valence_emmeans_stimulus_valence.csv")
save_emm(val_emms$valence$contrasts,
         "DataEMMs/Valence_negative_vs_neutral.csv")
save_emm(val_emms$within_pairs,
         "DataEMMs/Valence_pairwise_within_valence.csv")
save_emm(Valence_cell_pairs,
         "DataEMMs/Valence_pairwise_valence_by_behaviour_category.csv")

# Behaviour model
save_emm(beh_emms$cells,
         "DataEMMs/behaviour_emmeans_observed_valence_by_behaviour_category.csv")
save_emm(beh_emms$group$emmeans,
         "DataEMMs/behaviour_emmeans_participant_group.csv")
save_emm(beh_emms$group$contrasts,
         "DataEMMs/behaviour_pairwise_participant_group.csv")
save_emm(beh_emms$type$emmeans,
         "DataEMMs/behaviour_emmeans_stimulitype.csv")
save_emm(beh_emms$type$contrasts,
         "DataEMMs/behaviour_pairwise_stimulitype.csv")
save_emm(beh_emms$valence$emmeans,
         "DataEMMs/behaviour_emmeans_stimulus_valence.csv")
save_emm(beh_emms$valence$contrasts,
         "DataEMMs/behaviour_negative_vs_neutral.csv")
save_emm(beh_emms$within_pairs,
         "DataEMMs/behaviour_pairwise_within_valence.csv")
save_emm(behaviour_cell_pairs,
         "DataEMMs/behaviour_pairwise_valence_by_behaviour_category.csv")
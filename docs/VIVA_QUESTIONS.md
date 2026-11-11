# Viva Questions & Answers

**1. What is ensemble learning?** Combining several models so the group predicts better than any single one. It reduces variance (bagging), bias (boosting) or both.

**2. Bagging vs boosting?** Bagging trains trees in parallel on bootstrap samples and averages them (Random Forest). Boosting trains models sequentially, each focusing on the previous model's mistakes (AdaBoost, Gradient Boosting).

**3. What is CART?** Classification and Regression Trees: binary trees built by greedy splits that minimise Gini impurity (classification) or squared error (regression). Pruning (max_depth, min_samples_leaf) controls overfitting. It is the base learner of all our ensembles.

**4. How does AdaBoost work?** Start with equal sample weights; fit a weak learner; raise the weights of misclassified samples; give the learner a vote alpha = 0.5*ln((1-err)/err); repeat. Final prediction is the weighted vote.

**5. How does Gradient Boosting differ?** Each new tree fits the negative gradient (residual) of the loss, i.e. gradient descent in function space, scaled by a learning rate.

**6. What is an adaptive basis function model?** f(x) = w0 + sum_m w_m * phi_m(x; v_m) where the basis functions phi_m (with their own parameters v_m) are learned from data. Trees, boosting and neural networks are all examples.

**7. What is Bayesian model averaging (BMA)?** p(y|x,D) = sum_k p(y|x,M_k,D) p(M_k|D): average models weighted by posterior probability instead of choosing one. Here the weights are approximated as exp(-n_eff * validation log-loss), normalised.

**8. BMA vs stacking vs voting?** Voting: fixed (equal) weights. BMA: weights = posterior model probabilities, assumes the true model is in the set. Stacking: a meta-learner learns how to combine the outputs.

**9. What are Generalized Additive Models?** g(E[y]) = b0 + f1(x1) + ... + fp(xp) with smooth, learned f_j; more flexible than linear models but still interpretable. Boosting with depth-1 trees gives an additive model.

**10. Why is accuracy a poor metric here?** Only ~9% of transactions are fraud; predicting "legit" always gives ~91% accuracy. We use precision, recall, F1 and ROC-AUC.

**11. How did you handle class imbalance?** Stratified split plus class_weight='balanced' (CART, RF, AdaBoost trees) and balanced sample weights (Gradient Boosting).

**12. Precision vs recall in fraud?** Recall = share of frauds caught; precision = share of alerts that are real fraud. Missing fraud is costly, false alarms annoy customers; the 0.5 threshold can be tuned.

**13. Why soft voting?** It averages predicted probabilities, using the models' confidence, and is usually better than hard (majority) voting.

**14. Why is the dataset synthetic?** Real card data is confidential (and PCA-anonymised in public sets). Our generator embeds realistic fraud risk factors so models have signal. Results are illustrative, not production figures.

**15. Which features matter most?** Distance from home, amount relative to the customer's average, amount, distance from last transaction (see feature-importance chart).

**16. How would you improve it?** Real data, SMOTE/cost-sensitive learning, threshold tuning, XGBoost/LightGBM, stacking, drift monitoring, explainability (SHAP).

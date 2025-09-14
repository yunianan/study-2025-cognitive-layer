import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as sp
import plotly.express as px

def readable_pvalue(p):
    if p < .001:
        return f"{p:.2e}***"
    elif p < .01:
        return f"{p:.3f}**"
    elif p < .05:
        return f"{p:.3f}*"
    elif p >= 1:
        return "> .999"
    else:
        return f"{p:.2f}"

def print_title(title):
    print("===================================================================================================")
    print(title.upper().replace('_',' '))
    print("===================================================================================================")

def zscore(vec):
    return ((vec - vec.mean()) / vec.std()).astype(float)

def format_anova_table(model_fit):
    anova_table = sm.stats.anova_lm(model_fit, typ=2)
    anova_table['p'] = anova_table['PR(>F)'].apply(readable_pvalue)
    anova_table['partial_n2'] = anova_table.drop('Residual')['sum_sq'] / (anova_table.drop('Residual')['sum_sq']+anova_table.loc['Residual','sum_sq'])
    return anova_table

def permuted_anova(model_df,model_fit,n_permutations=1000):

    anova_table = format_anova_table(model_fit)
    perm_stats = {param: [] for param in anova_table.index if param != 'Residual'}

    formula = model_fit.model.formula
    outcome = formula.split('~')[0].strip()
    new_formula = formula.replace(f"{outcome} ~ ","outcome_shuffled ~ ")

    print(f"Permuting {n_permutations} times...")
    for i in range(n_permutations):
        model_df['outcome_shuffled'] = model_df[outcome].copy().sample(frac=1, random_state=i).values
        perm_model = smf.ols(new_formula, data=model_df).fit()
        perm_table = sm.stats.anova_lm(perm_model, typ=2)
        for param in perm_stats.keys():
            perm_stats[param].append(perm_table.loc[param,'F'])
    for param in perm_stats.keys():
        k = np.sum(np.asarray(perm_stats[param]) >= anova_table.loc[param, 'F'])
        anova_table.loc[param, 'PR(>F)'] = (k + 1) / (n_permutations + 1)
    
    anova_table['p'] = anova_table['PR(>F)'].apply(readable_pvalue)
    anova_table['partial_n2'] = anova_table.drop('Residual')['sum_sq'] / (anova_table.drop('Residual')['sum_sq']+anova_table.loc['Residual','sum_sq'])
    return anova_table

def anova_pairwise_comparisons(model_fit):
    contrasts = {
        "cognitive layer vs standalone LLMs": "C(condition)[T.standalone_llms]",
        "cognitive layer vs human therapist": "C(condition)[T.human_therapist]",
        "standalone LLMs vs human therapist": "C(condition)[T.human_therapist] - C(condition)[T.standalone_llms]"
    }
    rows = []
    for label, hyp in contrasts.items():
        res = model_fit.t_test(hyp)
        rows.append({
            'Contrast': label,
            'Adj. mean diff': res.effect[0],
            'SE': res.sd[0],
            't': res.tvalue[0],
            'df': model_fit.df_resid,
            'p': res.pvalue,
        })
    contrasts_df = pd.DataFrame(rows)
    contrasts_df['p _bonf'] = contrasts_df['p']*contrasts_df.shape[0]
    contrasts_df['p'] = contrasts_df['p'].apply(readable_pvalue)
    contrasts_df['p _bonf'] = contrasts_df['p _bonf'].apply(readable_pvalue)
    contrasts_df["Cohen's d"] = contrasts_df['Adj. mean diff'] / np.sqrt(model_fit.mse_resid)
    return contrasts_df

def display_residuals(model_fit,model_type='linear'):

    if model_type=='linear':
        residuals = model_fit.resid

        fig = px.histogram(residuals,title='Residuals',width=300,height=300)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title='Residuals')
        fig.show()

        shapiro_stat, shapiro_p = sp.shapiro(residuals)
        ks_stat, ks_p = sp.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
        print(f"Shapiro-Wilk: {shapiro_stat:.3f}, p = {readable_pvalue(shapiro_p)}")
        print(f"Kolmogorov-Smirnov: {ks_stat:.3f}, p = {readable_pvalue(ks_p)}")

        return bool((shapiro_p>=.05) | (ks_p>=.05))

    elif model_type=='logistic':
        pred_probs = model_fit.predict()
        bins = np.linspace(0, 1, 31)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        bin_indices = np.digitize(pred_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_labels)-1)
        binned = pd.DataFrame({
            'pred_prob': pred_probs,
            'residual': model_fit.resid_response,
            'bin': [i for i in bin_indices]
        })
        binned_stats = binned.groupby('bin').agg(
            mean_pred_prob=('pred_prob', 'mean'),
            mean_residual=('residual', 'mean'),
            count=('residual', 'size')
        ).reset_index()
        fig = px.scatter(
            binned_stats,
            x='bin',
            y='mean_residual',
            title='Binned Residuals (Logistic Regression)',
            labels={'bin': 'Predicted Probability Bin', 'mean_residual': 'Mean Residual'},
            hover_data=['mean_pred_prob', 'count'],
            width=400,
            height=300,
        )
        fig.update_layout(showlegend=False)
        fig.show()

def retrieve_analysed_data(selection='all'):

    df_pids = pd.read_csv('../data/study1_condition_allocation.csv')
    df_general = pd.read_csv('../data/study1_general_clinical.csv')

    if (selection=='all') | (selection=='clinicians'):
        df_ctrs = pd.read_csv('../data/study1_ctrs.csv')
        df_comparisons = pd.read_csv('../data/study1_pairwise_comparisons.csv')
    if (selection=='all') | (selection=='users'):
        df_users = pd.read_csv('../data/study1_user_ratings.csv')

    remove_index = (df_general['item']=='cbt_appropriate') & (df_general['score']=='Strong disagree')
    remove_pids = df_pids.loc[remove_index,'pid'].values
    print(f"Removing {len(remove_pids)} transcripts that are not appropriate for CBT")

    df_pids = df_pids[~df_pids['pid'].isin(remove_pids)].reset_index(drop=True)
    if (selection=='all') | (selection=='clinicians'):
        df_ctrs = df_ctrs[~df_ctrs['pid'].isin(remove_pids)].reset_index(drop=True)
        df_general = df_general[~df_general['pid'].isin(remove_pids)].reset_index(drop=True)
        df_comparisons = df_comparisons[(~df_comparisons['pid1'].isin(remove_pids))&(~df_comparisons['pid2'].isin(remove_pids))].reset_index(drop=True)
    if (selection=='all') | (selection=='users'):
        df_users = df_users[~df_users['pid'].isin(remove_pids)].reset_index(drop=True)

    N = len(df_pids)
    print(f"N = {N}")

    display(df_pids.groupby(['condition','model']).size())

    if selection=='all':
        return df_pids, df_ctrs, df_general, df_comparisons, df_users
    if selection=='clinicians':
        return df_pids, df_ctrs, df_general, df_comparisons
    elif selection=='users':
        return df_pids, df_users
    else:
        raise ValueError(f"Invalid data selection: {selection}")
    
def retrieve_realworld_data():
    df_rw_pids = pd.read_csv('../data/study2_realworld_users.csv')
    print(f"N users = {df_rw_pids['pid'].nunique()}")
    print(f"N transcripts = {df_rw_pids['conversation_id'].nunique()}")

    df_rw_ctrs = pd.read_csv('../data/study2_realworld_ctrs.csv')
    print(f"N transcripts with autograded CTRS scores = {df_rw_ctrs['conversation_id'].nunique()}")

    df_rw_clinical = pd.read_csv('../data/study2_realworld_clinical.csv')
    print(f"N users with clinical outcomes = {df_rw_clinical['pid'].nunique()}")

    df_rw_human_labels = pd.read_csv('../data/study2_realworld_human_labels.csv')
    print(f"N transcripts with human labels = {df_rw_human_labels['conversation_id'].nunique()}")

    df_rw_feedback = pd.read_csv('../data/study2_realworld_feedback.csv')
    print(f"N transcripts with feedback = {df_rw_feedback['conversation_id'].nunique()}")

    return df_rw_pids, df_rw_ctrs, df_rw_clinical, df_rw_human_labels, df_rw_feedback

def compute_icc(data_mat, icc_type='icc2'):
    """
    Compute ICC for a 2-column matrix (subjects x raters).
    icc_type: 'icc1', 'icc1k', 'icc2', 'icc2k', 'icc3', 'icc3k'
    """
    n, k = data_mat.shape
    mean_per_target = np.mean(data_mat, axis=1)
    mean_per_rater = np.mean(data_mat, axis=0)
    grand_mean = np.mean(data_mat)

    # Sums of squares
    ss_total = ((data_mat - grand_mean) ** 2).sum()
    ss_between = k * ((mean_per_target - grand_mean) ** 2).sum()
    ss_within = ((data_mat - mean_per_target[:, None]) ** 2).sum()
    ss_rater = n * ((mean_per_rater - grand_mean) ** 2).sum()
    ss_residual = ss_within - ss_rater

    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    ms_rater = ss_rater / (k - 1) if k > 1 else 0
    ms_residual = ss_residual / ((n - 1) * (k - 1)) if k > 1 else 0

    if icc_type == 'icc1':
        # One-way random, single rater
        return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    elif icc_type == 'icc1k':
        # One-way random, mean of k raters
        return (ms_between - ms_within) / ms_between
    elif icc_type == 'icc2':
        # Two-way random, single rater, absolute agreement
        return (ms_between - ms_residual) / (ms_between + (k - 1) * ms_residual + k * (ms_rater - ms_residual) / n)
    elif icc_type == 'icc2k':
        # Two-way random, mean of k raters, absolute agreement
        return (ms_between - ms_residual) / (ms_between + (ms_rater - ms_residual) / n)
    elif icc_type == 'icc3':
        # Two-way mixed, single rater, consistency
        return (ms_between - ms_residual) / (ms_between + (k - 1) * ms_residual)
    elif icc_type == 'icc3k':
        # Two-way mixed, mean of k raters, consistency
        return (ms_between - ms_residual) / ms_between
    else:
        raise ValueError("Unknown ICC type")

def compute_all_icc(df):
    icc_data = df.copy().dropna().to_numpy()

    icc_types = {
        'ICC(1,1)': 'icc1',
        'ICC(1,k)': 'icc1k',
        'ICC(2,1)': 'icc2',
        'ICC(2,k)': 'icc2k',
        'ICC(3,1)': 'icc3',
        'ICC(3,k)': 'icc3k'
    }

    print("Intraclass Correlation Coefficients (ICC) between human and classifier:")
    for label, icc_type in icc_types.items():
        icc_val = compute_icc(icc_data, icc_type=icc_type)
        print(f"{label}: {icc_val:.3f}")

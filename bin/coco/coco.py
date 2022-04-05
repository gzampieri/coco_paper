
import numpy as np
import pandas as pd
from cobra.core.gene import parse_gpr
from ast import Name, And, Or, BoolOp, Expression

MIN_EXP_SCALE = 10


class CoCo():
    """Main class for COndition-specific COmmunity model creation."""


    def __init__(self, gene_expr, default_ub=1000.):
        """Initialise data for CoCo model creation.

        Parameters
        ----------
        gene_expr : pandas DataFrame
            Table containing all the input transcriptomic profiles for T taxa over S samples.
        default_ub : float
            Default "infinity" flux upper bound of input models (usually 1000.).

        """

        self.gene_expr = gene_expr
        self.default_ub = default_ub
        # Correct null expression values using 1/MIN_EXP_SCALE of minimum expression value for each gene
        for i in range(self.gene_expr.shape[0]):
            self.gene_expr.iloc[i, :] = self.gene_expr.iloc[i, :].replace(0, \
                self.gene_expr.iloc[i, :].replace(0, self.gene_expr.iloc[i, :].max()).min() / MIN_EXP_SCALE)
        self.fold_changes = self.calculate_fold_changes(self.gene_expr)
        self.taxa = np.unique(self.fold_changes.index.get_level_values(0))
        self.samples = np.array(self.fold_changes.columns)
        self.alpha_matrix = pd.DataFrame(index=self.taxa, columns=self.samples)
        self.count_log_matrix = pd.DataFrame(index=self.taxa, columns=self.samples)
        self.__set_params() 


    def __set_params(self):
        """Calculate private parameters.
        
        These include a TxS matrix for the logarithmic map relaxation 
        and another TxS matrix for setting MAG-specific base bounds."""

        for s in self.samples:
            # total count sum for each MAG in a sample
            count_sums = self.gene_expr[s].sum(axis=0, level=0)
            # maximum count sum over all MAGs in the sample
            max_count = count_sums.max()
            # scale alphas based on the maximum count sum
            self.alpha_matrix.loc[:, s] = count_sums / max_count
            # set a logarithmic scale for MAG-specific base bounds
            self.count_log_matrix.loc[:, s] = np.log(count_sums + 1)
        return


    def get_alpha(self):
        return self.alpha_matrix


    def get_fold_changes(self):
        return self.fold_changes


    def get_count_logs(self):
        return self.count_log_matrix


    def get_param_range(self):
        """Estimate feasible parameter range."""
        
        max_factors = pd.DataFrame(index=self.taxa, columns=self.samples)
        for s in max_factors.columns:
            for t in max_factors.index:
                max_factors.loc[t, s] = np.log(self.fold_changes.loc[t, s].max())
        max_bounds = self.count_log_matrix.mul(max_factors) 
        return self.default_ub/max_bounds.max().max(), self.default_ub/self.count_log_matrix.max().max()


    def calculate_fold_changes(self, gene_expr):
        """Calculate gene expression fold change for each sample.

        Parameters
        ----------
        gene_expr : pandas DataFrame
            Table containing all the input transcriptomic profiles.

        Returns
        -------
        pandas DataFrame
            Table containing transcriptomic fold change profiles for all the samples.

        """

        num_samples = len(gene_expr.columns)
        expr_mean = pd.concat([gene_expr.mean(axis=1)]*num_samples, axis=1)
        expr_mean.columns = gene_expr.columns
        fold_changes = gene_expr.div(expr_mean)
        return fold_changes


    def build(self, model, sample_name, delta=1.0, gamma=1.0, meta=True):
        """Main function for COndition-specific COmmunity model creation.

        Parameters
        ----------
        model : micom Community instance
            Base community model.
        sample_name : string
            Name of the sample (i.e. condition).
        delta : float
            Parameter that controls the base bounds for each MAG model.
        gamma : float
            Parameter that controls the impact of gene expression on reaction bounds.
        meta : bool
            Whether or not to use the MAG-specific parametrisation (true for metatranscriptomics,
            false for standard transcriptomics as in METRADE)

        Returns
        -------
        micom Community instance
            A condition-specific community model.

        """

        if gamma <= 0:
            raise('gamma should be strictly positive!')
        if delta <= 0:
            raise('delta should be strictly positive!')
        
        rxns = pd.Series([r.id for r in model.reactions])
        ubs = np.array([r.upper_bound for r in model.reactions])
        lbs = np.array([r.lower_bound for r in model.reactions])
        gprs = np.array([r.gene_reaction_rule for r in model.reactions])

        # select gene expression values for the specified condition
        fold_change_dict = dict(zip(self.fold_changes.index.get_level_values(1), self.fold_changes[sample_name].values))
        
        # populate with 1s the fold changes for the genes in the model without available expression data
        gene_ids = [g.id for g in model.genes]
        missing_genes = pd.Series(gene_ids)[~pd.Series(gene_ids).isin(self.fold_changes.index.get_level_values(1))]
        fold_change_dict.update(dict(zip(missing_genes, [1.0]*len(missing_genes))))

        # calculate gene set expression (effective reaction expression)
        rxn_expr = np.array([1.0]*len(rxns))
        for i in range(len(rxn_expr)):
            rxn_expr[i] = self.__gene_set_expression(model.reactions[i].gene_reaction_rule, fold_change_dict)
        
        factors = np.array([1.0]*len(rxns))
        if meta:
            # calculate MAG-dependent bound factors (alpha and delta) for each reaction
            alphas = np.array([1.0]*len(rxns))
            deltas_up = ubs
            deltas_low = lbs
            idx1 = ubs == self.default_ub
            idx2 = lbs == -self.default_ub
            idx3 = gprs != ''
            for t in self.taxa:
                alphas[rxns.str.contains(t)] = self.alpha_matrix.loc[t, sample_name]
                deltas_up[np.logical_and.reduce([rxns.str.contains(t), idx1, idx3])] = delta * self.count_log_matrix.loc[t, sample_name]
                deltas_low[np.logical_and.reduce([rxns.str.contains(t), idx2, idx3])] = -delta * self.count_log_matrix.loc[t, sample_name]
            # calculate gene-dependent bound factors for each reaction
            idx1 = rxn_expr >= 1
            idx2 = rxn_expr < 1
            factors[idx1] = alphas[idx1] * (1 + gamma * alphas[idx1] * np.log(rxn_expr[idx1])) + (1.0 - alphas[idx1])
            factors[idx2] = alphas[idx2] / (1 + gamma * alphas[idx2] * np.abs(np.log(rxn_expr[idx2]))) + (1.0 - alphas[idx2])
            # apply new constraints
            for i in range(len(rxns)):
                model.reactions.get_by_id(rxns[i]).upper_bound = deltas_up[i] * factors[i]
                model.reactions.get_by_id(rxns[i]).lower_bound = deltas_low[i] * factors[i]
        else: # METRADE
            # calculate gene-dependent bound factors for each reaction
            idx1 = rxn_expr >= 1
            idx2 = rxn_expr < 1
            factors[idx1] = 1 + gamma * np.log(rxn_expr[idx1])
            factors[idx2] = 1 / (1 + gamma * np.abs(np.log(rxn_expr[idx2])))
            # apply new constraints
            for i in range(len(rxns)):
                model.reactions.get_by_id(rxns[i]).upper_bound = ubs[i] * factors[i]
                model.reactions.get_by_id(rxns[i]).lower_bound = lbs[i] * factors[i]
        
        return


    def __evaluate_gpr(self, expr, conf_genes):
        """Internal Corda-style evaluation of a gene-protein-reaction rule.
        
        Modified from https://github.com/resendislab/corda/blob/master/corda/util.py"""
        if isinstance(expr, Expression):
            return self.__evaluate_gpr(expr.body, conf_genes)
        elif isinstance(expr, Name):
            if expr.id not in conf_genes:
                return 1
            return conf_genes[expr.id]
        elif isinstance(expr, BoolOp):
            op = expr.op
            if isinstance(op, Or):
                return max(self.__evaluate_gpr(i, conf_genes) for i in expr.values)
            elif isinstance(op, And):
                return min(self.__evaluate_gpr(i, conf_genes) for i in expr.values)
            else:
                raise TypeError("unsupported operation " + op.__class__.__name__)
        elif expr is None:
            return 1
        else:
            raise TypeError("unsupported operation  " + repr(expr))


    def __gene_set_expression(self, rule, conf_genes):
        """Calculate effective reaction expression based on a gene-protein-reaction rule.

        Parameters:
        ----------
        rule : str
            A gene-protein-reaction rule. For instance "A and B" or "A or B".
        conf_genes : dict
            A str->float map denoting the mapping of gene IDs to expression values.
            
        Returns
        -------
        float
            Gene set expression.
        
        Modified from https://github.com/resendislab/corda/blob/master/corda/util.py
        """

        ast_rule, _ = parse_gpr(rule)
        return self.__evaluate_gpr(ast_rule, conf_genes)


    def __str__(self):
        return "A condition-specific community genome-scale metabolic model (CoCo-GEM) builder."

import numpy as np
import pandas as pd

BA_PURITY = 100

def set_medium(model, sample, medium_names, media_table, BA_lb):
    """Set medium constraints.

    Parameters
    ----------
    model : micom Community instance
        Community model.
    medium_name : str
        Medium ID.
    media_table : str
        Table with all media composition.
    BA_lb : float
        Lower bound for BA medium compounds (absolute value).
    YE_lb : float
        Lower bound for yeast extract and other compounds (absolute value).

    Returns
    -------
    micom Community instance
        Community model with new exchange constraints.

    """
    
    # set the purity of the BA
    YE_lb = BA_lb/BA_PURITY
    # set BA medium metabolites
    excRxns = [r.id for r in model.exchanges]
    idx = media_table['medium'].isin([medium_names[sample]])
    BAexcRxns = media_table['exchange_reaction'][np.logical_and(idx, media_table['exchange_reaction'].isin(excRxns))]
    d = dict(zip(BAexcRxns, [BA_lb]*len(BAexcRxns)))
    # set yeast extract
    idx = media_table['medium'].isin(['LCS'])
    LCSexcRxns = media_table['exchange_reaction'][np.logical_and(idx, media_table['exchange_reaction'].isin(excRxns))]
    YEexcRxns = pd.Series(excRxns)[np.logical_and(~pd.Series(excRxns).isin(BAexcRxns), ~pd.Series(excRxns).isin(LCSexcRxns))]
    d.update(dict(zip(YEexcRxns, [YE_lb]*len(YEexcRxns))))
    # adjust amino acids
    AAexcRxns = ['EX_ala__L_e_m', \
        'EX_arg__L_e_m',
        'EX_asn__L_e_m',
        'EX_asp__L_e_m',
        'EX_gln__L_e_m',
        'EX_glu__L_e_m',
        'EX_gly_e_m',
        'EX_his__L_e_m',
        'EX_ile__L_e_m',
        'EX_leu__L_e_m',
        'EX_lys__L_e_m',
        'EX_met__L_e_m',
        'EX_phe__L_e_m',
        'EX_pro__L_e_m',
        'EX_ser__L_e_m',
        'EX_thr__L_e_m',
        'EX_trp__L_e_m',
        'EX_tyr__L_e_m',
        'EX_val__L_e_m']
    d.update(dict(zip(AAexcRxns, [2*YE_lb]*len(YEexcRxns))))
    d['EX_cys__L_e_m'] = 0.0
    # block oxygen and key inputs/outputs
    d['EX_o2_e_m'] = 0.0
    d['EX_ch4_e_m'] = 0.0
    d['EX_co2_e_m'] = 0.0
    d['EX_h2_e_m'] = 0.0
    d['EX_co_e_m'] = 0.0

    model.medium = d

    return model


def set_biochemical_constraints(model, sample, biochemistry_table, set_biogas):
    """Set biochemical measurement constraints.

    Parameters
    ----------
    model : micom Community instance
        Community model.
    sample : str
        Sample name.
    biochemistry_table : pandas DataFrame
        Table with all the biochemical data.
    set_biogas : bool
        Whether to set the exchange rates for methane and carbon dioxide.

    Returns
    -------
    micom Community instance
        Community model with new exchange constraints.

    """
    
    rxns = [r.id for r in model.reactions]
    BH = biochemistry_table.columns[0:3]
    SAH = biochemistry_table.columns[3:6]
    AH = biochemistry_table.columns[6:]
    # set main inputs: acetate
    if sample == SAH[0] or sample == SAH[2]:
        model.reactions.EX_ac_e_m.lower_bound = biochemistry_table.loc['ac', sample] * 0.8
        model.reactions.EX_ac_e_m.upper_bound = biochemistry_table.loc['ac', sample] * 0.5
    elif sample == SAH[1]:
        model.reactions.EX_ac_e_m.lower_bound = biochemistry_table.loc['ac', sample]
        model.reactions.EX_ac_e_m.upper_bound = biochemistry_table.loc['ac', sample] * 0.8
    else:
        model.reactions.EX_ac_e_m.lower_bound = biochemistry_table.loc['ac', sample] - biochemistry_table.loc['ac_std', sample]
        model.reactions.EX_ac_e_m.upper_bound = biochemistry_table.loc['ac', sample] + biochemistry_table.loc['ac_std', sample]
    # hydrogen
    if sample in BH:
        model.reactions.EX_h2_e_m.lower_bound = 0.0
        model.reactions.EX_h2_e_m.upper_bound = 0.0
    else:
        model.reactions.EX_h2_e_m.lower_bound = biochemistry_table.loc['h2', sample] - biochemistry_table.loc['h2_std', sample]
        model.reactions.EX_h2_e_m.upper_bound = biochemistry_table.loc['h2', sample] + biochemistry_table.loc['h2_std', sample]
    # set main outputs:
    if set_biogas:
        # methane
        model.reactions.EX_ch4_e_m.lower_bound = biochemistry_table.loc['ch4', sample] - biochemistry_table.loc['ch4_std', sample]
        model.reactions.EX_ch4_e_m.upper_bound = biochemistry_table.loc['ch4', sample] + biochemistry_table.loc['ch4_std', sample]
        # carbon dioxide
        model.reactions.EX_co2_e_m.lower_bound = biochemistry_table.loc['co2', sample] - biochemistry_table.loc['co2_std', sample]
        model.reactions.EX_co2_e_m.upper_bound = biochemistry_table.loc['co2', sample] + biochemistry_table.loc['co2_std', sample]
    # set other monitored metabolites
    if 'EX_etoh_e_m' in rxns: # ethanol
        model.reactions.EX_etoh_e_m.upper_bound = 0
    if 'EX_ppoh_e_m' in rxns: # 1-propanol
        model.reactions.EX_ppoh_e_m.upper_bound = 0
    if 'EX_btoh_e_m' in rxns: # 1-butanol
        model.reactions.EX_btoh_e_m.upper_bound = 0
    if 'EX_1btol_e_m' in rxns: # alternative 1-butanol
        model.reactions.EX_1btol_e_m.upper_bound = 0
    if 'EX_iamoh_e_m' in rxns: # iso-amylalcohol
        model.reactions.EX_iamoh_e_m.upper_bound = 0
    if 'EX_ibt_e_m' in rxns: # isobutyrate
        model.reactions.EX_ibt_e_m.upper_bound = 0
    if 'EX_isobuta_e_m' in rxns: # alternative isobutyrate
        model.reactions.EX_isobuta_e_m.upper_bound = 0
    if 'EX_but_e_m' in rxns: # butyrate
        if sample in BH or sample in AH:
            model.reactions.EX_but_e_m.upper_bound = 0
    if 'EX_ival_e_m' in rxns: # isovalerate
        if sample in BH or sample in AH:
            model.reactions.EX_ival_e_m.upper_bound = 0
    if 'EX_pta_e_m' in rxns: # valerate
        model.reactions.EX_pta_e_m.upper_bound = 0
    if 'EX_hxa_e_m' in rxns: # 1-hexanoate
        model.reactions.EX_hxa_e_m.upper_bound = 0
    if 'EX_caproic_e_m' in rxns: # alternative 1-hexanoate
        model.reactions.EX_caproic_e_m.upper_bound = 0

    return model


def set_MAG_constraints(model):
    hydrogenotrophic_species = ['bin_26_operams', 'bin_23_metaspades', 'bin_57_operams', 'bin_25_operams']
    for s in hydrogenotrophic_species:
        if s in model.taxa:
            model.reactions.get_by_id('EX_co2_e__' + s).upper_bound = 0.0
            model.reactions.get_by_id('EX_h2_e__' + s).upper_bound = 0.0
    return model


def relax_MAG_constraints(model):
    hydrogenotrophic_species = ['bin_26_operams', 'bin_23_metaspades', 'bin_57_operams', 'bin_25_operams']
    for s in hydrogenotrophic_species:
        if s in model.taxa:
            model.reactions.get_by_id('EX_co2_e__' + s).upper_bound = 1.
            model.reactions.get_by_id('EX_h2_e__' + s).upper_bound = 1.
    return model

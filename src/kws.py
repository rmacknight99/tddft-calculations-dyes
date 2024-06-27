import autode as ade

import autode as ade

# Common Keywords for Geometry Optimization and TD-DFT Calculations
common_kws = [
    'PBE0', 
    'def2-TZVP',
    'RIJCOSX',
    'D4'
]

# Ground-State Geometry Optimization
opt_kws = ade.wrappers.keywords.OptKeywords(['Opt'] + common_kws)

# Common TD-DFT Keywords for Both Absorption and Emission Spectra
tddft_kws = ade.wrappers.keywords.SinglePointKeywords(common_kws)

# TD-DFT Block
tddft_block = """
%tddft
  nroots 20
  maxdim 5
end
"""

# Excited-State Geometry Optimization
exc_opt_block = """
%tddft
  nroots 1
  IRoot 1
end
"""

# STEOM-DLPNO-CCSD for Excited States Calculation
steom_kws = ade.wrappers.keywords.SinglePointKeywords([
    'STEOM-DLPNO-CCSD', 
    'def2-TZVP', 
    'def2-TZVP/C',
    'RIJCOSX',
    'CPCM(HEXANE)'
])

steom_block = """
%mdci
  nroots 20
  dosolv true
end
"""
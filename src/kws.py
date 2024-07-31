import autode as ade

server = "'http://gpg-boltzmann.cheme.cmu.edu:5002/'"
G16_external = f'External="aimnetg16 --server {server}"'

ORCA_KWS = {}

GAUSSIAN_KWS = {}

ORCA_KWS['opt'] = ade.wrappers.keywords.OptKeywords(['Opt', 'PBE0', 'def2-TZVP', 'RIJCOSX', 'D4'])
ORCA_KWS['exc_opt'] = ade.wrappers.keywords.OptKeywords(['Opt', 'PBE0', 'def2-TZVP', 'RIJCOSX', 'D4'])

GAUSSIAN_KWS['opt'] = ade.wrappers.keywords.OptKeywords(['Opt', G16_external])
GAUSSIAN_KWS['exc_opt'] = ade.wrappers.keywords.OptKeywords(['Opt', G16_external, 'TD(NStates=1, root=1)'])

ORCA_KWS['tddft'] = ade.wrappers.keywords.SinglePointKeywords(['PBE0', 'def2-TZVP', 'RIJCOSX', 'D4'])
ORCA_KWS['tddft_ccsd'] = ade.wrappers.keywords.SinglePointKeywords(['STEOM-DLPNO-CCSD', 'def2-TZVP', 'def2-TZVP/C', 'RIJCOSX'])

GAUSSIAN_KWS['tddft'] = ade.wrappers.keywords.SinglePointKeywords(['PBE0', 'def2-TZVP', 'd3bj', 'TD(NStates=10)'])

ORCA_KWS['blocks'] = {
  'tddft': """
%tddft
  nroots 20
  maxdim 5
end
""",
  'exc_opt': """
%tddft
  nroots 1
  IRoot 1
end
""",
  'steom': """
%mdci
  nroots 5
  dosolv false
end
"""
}

# Andersen Pasavento Wang
# theta_0 = 11.5 deg
# h = thickness = 0.081 cm
# l = h / \beta
# \beta = aspect ratio = 1/8
#  nu = 0.0089 cm^2 / s 
# rhos  rho_f = 2.7
# Re = sqrt(2 * 1.7 * 980.665 * 0.081) * 0.081*8 / 0.0089 ~= 1200
# ( The paper says 1025?? )
# We use Ga = sqrt(2 * 1.7 * 9.8 * 0.02) * 0.16 / 0.00011

OPTIONS="-bpdx 64 -bpdy 64 -tdump 0.01 -CFL 0.02 -extent 5 -nu 0.0089 -tend 80 -poissonType cosine -DLM 1 -iterativePenalization 0"
OBJECTS='ellipse semiAxisX=0.324 semiAxisY=0.0405 ypos=1.5 angle=11.5 bFixed=1 rhoS=2.7
'

source launchCommon.sh

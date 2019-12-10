# Around Reynolds 175? :
# Galileo number = sqrt(Delta Rho / Rho_f * g * D) * D / nu = 138
# term vel \approx sqrt(pi * R * (\rho*-1) g )
# Rho_s = 1.01 * Rho_f
NU=${NU:-0.0005548658217}
LAMBDA=${LAMBDA:-0}
BPDX=${BPDX:-32}
BPDY=${BPDY:-48}

OPTIONS=" -CFL 0.1 -DLM 1 -bpdx $BPDX -bpdy $BPDY -tdump 0.1 -nu ${NU} -extent 3 -tend 500 -poissonType cosine -iterativePenalization 0 -lambda ${LAMBDA}"
OBJECTS='disk radius=0.1 ypos=0.5 bFixed=1 rhoS=1.10
'

source launchCommon.sh

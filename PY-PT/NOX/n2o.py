from numpy import exp


def TSat(P):
    # Input in Pa
    P = P * 0.000145038
    T= -0.00000000000000150576*P**(6) + 0.00000000000532248688*P**(5) - 0.00000000749212638441*P**(4) + 0.00000539928454090928*P**(3) - 0.00217592516372610000*P**(2) + 0.57533258993215300000*P + 184.06671769203400000000
    return T

def PSat(T):
    # SEE <THERMOPHYSICAL PROPERTIES OF NITROUS OXIDE (IHS - ESDU)> FOR REF.

    # RANGE OF APPLICABILITY (IN CELCIUS) : -90.82 to 36.42 (182.33 to 309.57 K)

    # SUBSCRIPTS
        # c - value at critical point
        # r - denotes reduced quantity, e.g. T_r = T/T_c

    # CONSTANTS USED IN THE EQUATION(NO UNITS)

    b1 = -6.71893
    b2 = 1.35966
    b3 = -1.3779
    b4 = -4.051

    # PHYSICAL CONSTANTS OF NITROUS OXIDE USED IN THE EQUATION

    p_c = 7251 # Critical Pressure in kPa
    T_c = 309.57 # Critical Temperature in Kelvins --> 36.42 C
    T_r = T/T_c # Dimensionless

    # Equation 4.1 which gives the Vapour Pressure (in kPa) of N2O from:
    # <Thermophysical properties of nitrous oxide (IHS - ESDU)> (PDF)
    # Formula also used in:
    # <Modelling the nitrous run tank emptying (Aspire Space) (page 8)
    p_kPa = p_c * exp((1/T_r)*(b1*(1-T_r)+b2*(1-T_r)**(3/2)+b3*(1-T_r)**(5/2)+b4*(1-T_r)**5))
    P = p_kPa * 1000
    # Returns the saturation pressure (in Pa) at temperature T.
    return P

def rhoLiqSat(T):
    # SEE <THERMOPHYSICAL PROPERTIES OF NITROUS OXIDE (IHS - ESDU)> FOR REF.

    # RANGE OF APPLICABILITY (IN CELCIUS) : -90.82 to 36.42 (182.33 to 309.57 K)

    # SUBSCRIPTS
        # c - value at critical point
        # r - denotes reduced quantity, e.g. T_r = T/T_c

    # STATE DESCRIPTOR
        # l - value for saturated liquid (IT'S A SMALL 'L', NOT A '1' (ONE)) 

    # CONSTANTS USED IN THE EQUATION(NO UNITS)

    b1 = 1.72328
    b2 = -0.83950
    b3 = 0.51060
    b4 = -0.10412

    # PHYSICAL CONSTANTS OF NITROUS OXIDE USED IN THE EQUATION

    rho_c = 452 # Critical Density in kg/m^3
    T_c = 309.57 # Critical Temperature in Kelvins --> 36.42 C
    T_r = T/T_c # Dimensionless

    # Equation 4.2 which gives the Density (in kg/m^3) of the Saturated N2O Liquid from:
    # <Thermophysical properties of nitrous oxide (IHS - ESDU)> (PDF)
    # Formula also used in:
    # <Modelling the nitrous run tank emptying (Aspire Space) (page 8)
    rho_l = rho_c * exp(b1*(1-T_r)**(1/3)+b2*(1-T_r)**(2/3)+b3*(1-T_r)+b4*(1-T_r)**(4/3))
    return rho_l

def rhoVapSat(T):
    # SEE <THERMOPHYSICAL PROPERTIES OF NITROUS OXIDE (IHS - ESDU)> FOR REF.

    # RANGE OF APPLICABILITY (IN CELCIUS) : -90.82 to 36.42 (182.33 to 309.57 K)

    # SUBSCRIPTS
        # c - value at critical point
        # r - denotes reduced quantity, e.g. T_r = T/T_c

    # STATE DESCRIPTOR
        # g - value for gas or saturated vapour

    # CONSTANTS USED IN THE EQUATION(NO UNITS)

    b1 = -1.00900
    b2 = -6.28792
    b3 = 7.50332
    b4 = -7.90463
    b5 = 0.629427

    # PHYSICAL CONSTANTS OF NITROUS OXIDE USED IN THE EQUATION

    rho_c = 452 # Critical Density in kg/m^3
    T_c = 309.57 # Critical Temperature in Kelvins --> 36.42 C
    T_r = T/T_c # Dimensionless

    # Equation 4.3 which gives the Density (in kg/m^3) of the Saturated N2O Vapour from:
    # <Thermophysical properties of nitrous oxide (IHS - ESDU)> (PDF)
    # Formula also used in:
    # <Modelling the nitrous run tank emptying (Aspire Space) (page 8)
    rho_g = rho_c * exp(b1*((1/T_r)-1)**(1/3)+b2*((1/T_r)-1)**(2/3)+b3*((1/T_r)-1)+b4*((1/T_r)-1)**(4/3)+b5*((1/T_r)-1)**(5/3))
    return rho_g

def hLiqSat(T):
    # SEE <THERMOPHYSICAL PROPERTIES OF NITROUS OXIDE (IHS - ESDU)> FOR REF.

    # RANGE OF APPLICABILITY (IN CELCIUS) : -90.82 to 35 (182.33 to 308 K)

    # SUBSCRIPTS
        # c - value at critical point
        # r - denotes reduced quantity, e.g. T_r = T/T_c

    # STATE DESCRIPTOR
        # l - value for saturated liquid (IT'S A SMALL 'L', NOT A '1' (ONE))

    # CONSTANTS USED IN THE EQUATION(kJ/kg)

    b1 = -200
    b2 = 116.043
    b3 = -917.225
    b4 = 794.779
    b5 = -589.587

    # PHYSICAL CONSTANTS OF NITROUS OXIDE USED IN THE EQUATION

    T_c = 309.57 # Critical Temperature in Kelvins --> 36.42 C
    T_r = T/T_c # Dimensionless

    # Equation 4.4 which gives the Specific Enthalpy (in kJ/kg) of the Saturated N2O Liquid from:
    # <Thermophysical properties of nitrous oxide (IHS - ESDU)> (PDF)
    # Formula also used in:
    # <Modelling the nitrous run tank emptying (Aspire Space) (page 8)
    h_l = b1+b2*(1-T_r)**(1/3)+b3*(1-T_r)**(2/3)+b4*(1-T_r)+b5*(1-T_r)**(4/3)
    return h_l*1000    # Output in J/kg


def hVapSat(T):
    # SEE <THERMOPHYSICAL PROPERTIES OF NITROUS OXIDE (IHS - ESDU)> FOR REF.

    # RANGE OF APPLICABILITY (IN CELCIUS) : -90.82 to 35 (182.33 to 308 K)

    # SUBSCRIPTS
        # c - value at critical point
        # r - denotes reduced quantity, e.g. T_r = T/T_c

    # STATE DESCRIPTOR
        # g - value for gas or saturated vapour

    # CONSTANTS USED IN THE EQUATION(kJ/kg)

    b1 = -200
    b2 = 440.055
    b3 = -459.701
    b4 = 434.081
    b5 = -485.338


    # PHYSICAL CONSTANTS OF NITROUS OXIDE USED IN THE EQUATION

    T_c = 309.57 # Critical Temperature in Kelvins --> 36.42 C
    T_r = T/T_c # Dimensionless

    # Equation 4.4 which gives the Specific Enthalpy (in kJ/kg) of the Saturated N2O Liquid from:
    # <Thermophysical properties of nitrous oxide (IHS - ESDU)> (PDF)
    # Formula also used in:
    # <Modelling the nitrous run tank emptying (Aspire Space) (page 8)
    h_g = b1+b2*(1-T_r)**(1/3)+b3*(1-T_r)**(2/3)+b4*(1-T_r)+b5*(1-T_r)**(4/3)
    return h_g*1000    # Output in J/kg

def sLiqSat(T):
    # CALCULATES NITROUS OXIDE LIQUID SATURATION ENTROPY 
    # RANGE OF APPLICABILITY (IN KELVIN) : 200 to 309.5 K

    S_l = 0.00000000000315581972*T**6 - 0.00000000468566170810*T**5 + 0.00000288944492115364*T**4 - 0.00094705544528601000*T**3 + 0.17397838118169700000*T**2 - 16.97324460874480000000*T + 686.42428840194000000000
    return S_l*1000    # Output in J/K

def sVapSat(T):
    # CALCULATES NITROUS OXIDE VAPOR SATURATION ENTROPY 
    # RANGE OF APPLICABILITY (IN KELVIN) : 200 to 309.5 K

    S_v = -0.00000000000438756376*T**6 + 0.00000000650865606247*T**5 - 0.00000400997574354474*T**4 + 0.00131305477621871000*T**3 - 0.24095141158161400000*T**2 + 23.48325573855770000000*T - 947.12161466815600000000
    return S_v*1000    # Output in J/K

def CpLiqSat(T):
    # SEE <THERMOPHYSICAL PROPERTIES OF NITROUS OXIDE (IHS - ESDU)> FOR REF.

    # RANGE OF APPLICABILITY (IN CELCIUS) : -90.82 to 30 (182.33 to 303.15 K)

    # SUBSCRIPTS
        # c - value at critical point
        # r - denotes reduced quantity, e.g. T_r = T/T_c

    # STATE DESCRIPTOR
        # l - value for saturated liquid (IT'S A SMALL 'L', NOT A '1' (ONE))

    # CONSTANTS USED IN THE EQUATION

    b1 = 2.49973 # (kJ/kg)
    b2 = 0.023454 # (dimensionless)
    b3 = -3.80136 # (dimensionless)
    b4 = 13.0945 # (dimensionless)
    b5 = -14.5180 # (dimensionless)

    # PHYSICAL CONSTANTS OF NITROUS OXIDE USED IN THE EQUATION

    T_c = 309.57 # Critical Temperature in Kelvins --> 36.42 C
    T_r = T/T_c # Dimensionless

    # Equation 4.7 which gives the Isobaric Specific Heat Capacity (in kJ/kg * K) of the Saturated N2O Liquid from:
    # <Thermophysical properties of nitrous oxide (IHS - ESDU)> (PDF)
    # Formula also used in:
    # <Modelling the nitrous run tank emptying (Aspire Space) (page 8)
    c_p_l = b1*(1+b2*(1-T_r)**(-1)+b3*(1-T_r)+b4*(1-T_r)**2+b5*(1-T_r)**(3))
    return c_p_l*1000  # Output in J/kgK

def CpVapSat(T):
    # SEE <THERMOPHYSICAL PROPERTIES OF NITROUS OXIDE (IHS - ESDU)> FOR REF.

    # RANGE OF APPLICABILITY (IN CELCIUS) : -90.82 to 30 (182.33 to 303.15 K)

    # SUBSCRIPTS
        # c - value at critical point
        # r - denotes reduced quantity, e.g. T_r = T/T_c

    # STATE DESCRIPTOR
        # l - value for saturated liquid (IT'S A SMALL 'L', NOT A '1' (ONE))

    # CONSTANTS USED IN THE EQUATION
    # 132.632 0.052187 -0.364923 -1.20233 0.536141
    b1 = 132.632 # (kJ/kg)
    b2 = 0.052187 # (dimensionless)
    b3 = -0.364923 # (dimensionless)
    b4 = -1.20233 # (dimensionless)
    b5 = 0.536141 # (dimensionless)

    # PHYSICAL CONSTANTS OF NITROUS OXIDE USED IN THE EQUATION

    T_c = 309.57 # Critical Temperature in Kelvins --> 36.42 C
    T_r = T/T_c # Dimensionless

    # Equation 4.7 which gives the Isobaric Specific Heat Capacity (in kJ/kg * K) of the Saturated N2O Vapor from:
    # <Thermophysical properties of nitrous oxide (IHS - ESDU)> (PDF)
    # Formula also used in:
    # <Modelling the nitrous run tank emptying (Aspire Space) (page 8)
    c_p_l = b1*(1+b2*(1-T_r)**(-2/3)+b3*(1-T_r)**(-1/3)+b4*(1-T_r)**(1/3)+b5*(1-T_r)**(2/3))
    return c_p_l*1000 # Output in J/kgK


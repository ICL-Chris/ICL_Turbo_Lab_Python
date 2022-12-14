#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This modules contains common data conversion and calculation functions


Created on Mon Nov 15 11:14:55 2021

@author: cnoon
"""

class vCone():
    """Specifies an object for calculating velocity/massflow for a given vCone geometry
    
    """
    
    def __init__(self, Dpipe, Dcone, Cd = 0.82, **keywords):
        """Creates a new vCone object with a given geometry
        
        """
        
        self.parameters = {'U1': 0.001,
                           'U2': 1000000,
                            'U3': 100,
                            'U5': 348.338,
                            'k': 1.402,
                            'Z': 0.99721,
                            'Sg': 1,
                            'alphaPipe': 8.889e-6,
                            'alphaCone': 6.111e-6
                            }
        
        self.Dpipe = Dpipe
        self.Dcone = Dcone
        self.Cd = Cd
        self.Beta = (1 - (Dcone ** 2 / Dpipe ** 2)) ** 0.5
        
        for word in keywords:
            if word in self.parameters:
                self.parameters[word] = keywords[word]
            else:
                print('keyword "',word,'" not recognised')
            
    def volumeFlowRate(self, staticPressure, deltaPressure, temperature):
        """ calculate volume flowrate for the vcone for given pressure and temperature data
        
        static pressure is in Bar
        differential pressure is in mBar
        temperature is in Kelvin
        """
        from math import pi
        
        rho = (self.parameters['U5'] * self.parameters['Sg'] * 
               staticPressure) / ( self.parameters['Z'] * temperature )
        
        Td = (9.0 * temperature / 5.0 - 527.67)
        
        Fa = ((self.Dpipe ** 2.0 - self.Dcone ** 2 ) /
            (((1 - self.parameters['alphaPipe'] * Td ) * self.Dpipe)**2 -
            ((1 - self.parameters['alphaCone'] * Td ) * self.Dcone)**2 ))
        
        Y = (1 - ( 0.649 + 0.696 * self.Beta**4 ) * self.parameters['U1'] *
             deltaPressure / (self.parameters['k'] * staticPressure))
        
        k1 = (((pi * (2 * self.parameters['U3'])**0.5 ) / (4 * self.parameters['U2'])) *
            ((self.Dpipe**2 * self.Beta**2 ) / ( 1 - self.Beta**4 )**0.5 ))

        Qv = Fa * self.Cd * Y * k1 * ( deltaPressure / rho ) ** 0.5
        
        return Qv
    
    def massFlowRate(self, staticPressure, deltaPressure, temperature):
        """ calculate massflow rate for the Vcone for given pressure and temperature data
        
        static pressure is in Bar
        differential pressure is in mBar
        temperature is in Kelvin
        """
        
        rho = ((self.parameters['U5'] * self.parameters['Sg'] * staticPressure)
        / ( self.parameters['Z'] * temperature ))
        
        Qv = self.volumeFlowRate(staticPressure, deltaPressure, temperature)
        
        Mdot = Qv * rho
        
        return Mdot
from numpy import genfromtxt, meshgrid, log10, pi, linspace, array, zeros, log, exp, roots, sqrt, trapz
from scipy import interpolate
from center_energy import *
from matplotlib.pyplot import *
import matplotlib.font_manager as font_manager
import sys


'''
This code was developed in the AST3310 course (Astrophysical Plasma and Stellar Interiors),
and it models the inside of a star (fusion processes, heat transfer through radiation and convection,
density and temperature profiles and so on).
'''

# Set axis and title font and size:
title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} 
axis_font = {'fontname':'Arial', 'size':'20'}


class stellar_core:
	def __init__(self, R0_frac, T0, rho0_frac):
		'''
		Method that defines all the needed constants and parameters.
		'''

		self.rho_sun_avg = 1.408*10**3	    	# average density sun [kg*m**(-3)]
		self.L_sun       = 3.846*10**(26)       # luminosity sun [W]
		self.R_sun       = 6.96*10**8           # radius sun [m]
		self.M_sun       = 1.989*10**(30)		# mass sun [kg]

		self.L0    = 1.0*self.L_sun				# luminosity [W]
		self.M0    = 1.0*self.M_sun				# mass [kg]
		self.rho0  = rho0_frac*self.rho_sun_avg	# density [kg*m**(-3)]
		self.T0    = T0							# temperature [K]
		self.R0    = R0_frac*self.R_sun         # radius [m]

		X          = 0.7						# mass fraction H
		Y          = 0.29						# mass fraction He
		Y3         = 10**(-10)					# mass fraction He-3
		Z          = 0.01						# mass fraction metals
		Z_Li7      = 10**(-13)					# mass fraction Li-7
		Z_Be7      = 10**(-13)					# mass fraction Be-7

		self.k_B   = 1.382*10**(-23)			# Boltzmann constant [m**2*kg*s**(-2)*K**(-1)]
		self.m_u   = 1.6605*10**(-27)			# mass unit [kg]
		self.sigma = 5.67*10**(-8)		    	# W*m**(-2)*K**(-4)
		self.c     = 299792458                  # speed of light [m/s]
		self.G     = 6.67408*10**(-11)		    # Gravitational constant [m**3*kg**(-1)*s**(-2)]

		# average particle weight [unitless]:
		self.mu    = 1./(2.*X + 3.*(Y-Y3)/4. + Y3 + (Z-Z_Li7-Z_Be7)/2. + 4.*Z_Li7/7. + 5.*Z_Be7/7.)

		# from equation of state and convention:
		self.delta = 1.0													  			 
		self.alpha = 1.0

		# specific heat capacity (constant pressure) for ideal gas:
		self.C_p   = (5*self.k_B)/(2*self.mu*self.m_u)																  

	def opacity(self, rho, T):
		'''
		Method that calculate the opacity of the star by either picking out,
		interpolating or extrapolating the tabulated values in the file opacity.txt
		'''
		rho    = rho*10**(3)/10**(6)           # convert to cgs-units

		#read data from file:
		data   = genfromtxt('opacity.txt')       
		log10R = data[0,1:]                    # R = rho/(T/10**6)**3, [rho] = g/cm**3
		log10T = data[1:,0]                    # [T] = K
		log10kappa = data[1:,1:]               # [kappa] = cm**2/g

		logT   = log10(T)
		logR   = log10(rho/(T/10**6)**3)

		#find values:
		'''
		the R and T values already in the table will get their tabulated kappa values,
		and the R and T values in between will get interpolated kappa values. If T or 
		R is bigger or smaller than the tabulated values, the interpolate function extrapolates
		by giving the last/closest available kappa value.
		'''  
		f        = interpolate.interp2d(log10R,log10T,log10kappa, kind='linear', bounds_error=False) 
		logkappa = f(logR, logT)
		kappa    = (10**logkappa)*10**(-4)/10**(-3)  # converting to m**2/kg

		#let us know if kappa value has been extrapolated:
		if logT > max(log10T):
			print 'Temperature too high, opacity has been extrapolated'
		if logT < min(log10T):
			print 'Temperature too low, opacity has been extrapolated'
		if logR > max(log10R):
			print 'Density too high, opacity has been extrapolated'
		if logR < min(log10R):
			print 'Density too low, opacity has been extrapolated'

		return kappa

	def pressure(self, rho, T):
		'''
		Method that calculates the pressure, P = P_G + P_rad, for a given density and temperature.
		'''
		k_B, c         = self.k_B, self.c
		mu, m_u, sigma = self.mu, self.m_u, self.sigma

		P_G   = rho*k_B*T/(mu*m_u)       # gas pressure
		P_rad = 4*sigma*T**4/(3*c)       # radiational pressure
		P     = P_G + P_rad	             # kg*m**(-1)*s**(-2) = pascal

		return P 

	def density(self, P, T):
		'''
		Method that calculates the density for a given pressure and temperature.
		'''
		mu, m_u, k_B, c, sigma = self.mu, self.m_u, self.k_B, self.c, self.sigma

		P_rad = 4*sigma*T**4/(3*c)
		P_G   = P - P_rad   
		rho   = P_G*mu*m_u/(k_B*T)

		return rho

	def ODE_solution(self, dm1 = -1e27, dynamic=True):
		'''
		Method to solve the four differential equations to find the developement of the
		radius, pressure, luminosity and temperature as we move further inside the star.
		The first input is the step size, with a default of -1e27 kg.
		The Forward Euler method is used to solve the ODEs. The returned values are arrays 
		containing the values of r, P, L, rho, epsilon and T with respect to the mass 
		(how far inside the star we are), and also an index telling how far the calculation got.
		'''
		sigma, G, M0, L0, rho0, T0, R0 = self.sigma, self.G, self.M0, self.L0, self.rho0, self.T0, self.R0

		n = 10000							# numbers of steps in solving ODEs

		# physical properties:
		m   = zeros(n)						# array containing mass-values
		r   = zeros(n)						# array containing radius
		P   = zeros(n)						# array containing pressure
		L   = zeros(n)						# array containing luminosity
		T   = zeros(n)						# array containing temperature
		rho = zeros(n)						# array containing density

		# Energy production:
		epsilon = zeros(n)					# array containing epsilon = sum(rates x energy)
		PPI     = zeros(n)					# array containing energy produced by PPI
		PPII    = zeros(n)					# array containing energy produced by PPII
		E_tot   = zeros(n)					# array containing total energy produced 

		# Flux:
		F_r   = zeros(n)					# array containing radiative flux
		F_c   = zeros(n)					# array containing convective flux
		F_tot = zeros(n)					# array containing total flux

		# Gradients:
		nabla_adiabatic = zeros(n)
		nabla_stable    = zeros(n)
		nabla_star      = zeros(n)

		#set initial conditions:
		m[0]   = M0
		r[0]   = R0
		P[0]   = self.pressure(rho0, T0)
		L[0]   = L0
		T[0]   = T0
		rho[0] = rho0

		#Solving ODEs by Forward Euler Method:

		for i in range(n-1):
			# if-test stopping loop if mass has reached zero
			if m[i]<=0:
				index = i
				print 'Mass has reached zero, index =', i
				break
			elif r[i] < 0.001*R0:
				print 'r < 0.001 R0, index =', i
				break
			elif L[i] < 0.001*L0:
				print 'L < 0.001 L0, index=', i
				break
			# computing new values step by step:
			else:
				index = i

				center_energy_       = center_energy(T[i], rho[i])				# use class from Appendix C
				[epsil,pp1,pp2,Etot] = center_energy_.energy()					# get energy-values for given T and rho
				epsilon[i] = epsil 												# get epsilon-value for given T and rho
				PPI[i]     = pp1												# get energy produced by PPI for given T and rho
				PPII[i]    = pp2												# get energy produced by PPII for given T and rho
				E_tot[i]   = Etot 												# get total energy produced for given T and rho

				kappa = self.opacity(rho[i], T[i])								# find opacity for current rho and T
				p     = 1e-2													# fraction variables are allowed to change

				dr = 1./(4*pi*r[i]**2*rho[i])									# dr/dm, change in r for step i
				dP = -G*m[i]/(4*pi*r[i]**4)										# dP/dm, change in P for step i
				dL = epsilon[i]													# dL/dm, change in L for step i

				# gradients:
				nabla_adiabatic[i] = 2./5												 # adiabatic gradient
				nabla_stable[i]    = 3*L[i]*kappa*P[i]/(64*pi*self.G*sigma*m[i]*T[i]**4) # stable gradient
				
				# Check if layer is convectively stable and choose dT/dm:
				if nabla_stable[i] > nabla_adiabatic[i]:

					# constants needed to check if layer is stable:																  
					g   = (self.G*m[i])/r[i]**2												         # gravitational acceleration [m/s**2]
					H_p = P[i]*r[i]**2/(self.G*rho[i]*m[i])							  		         # Pressure scale height
					l_m = self.alpha*H_p													         # mixing length 
					U   = ((64*sigma*T[i]**3)/(3*kappa*rho[i]**2*self.C_p))*sqrt(H_p/(g*self.delta)) # paramener for cubic eq.
					
					
					# solve equation of third order:
					R = U/l_m**2
					K = 4*R

					# coefficients:
					a       = 1./R
					b       = 1.0
					c       = K
					d       = -(nabla_stable[i] - nabla_adiabatic[i])
					coeff   = [a,b,c,d]
					xi      = roots(coeff)
					xi_real = []

					# not interested in complex solutions, filter out complex roots:
					for j in range(len(xi)):
						if xi[j].imag == 0.0:
							xi_real.append(xi[j])

					# remove complex part of root: a + 0j
					xi_real_ = zeros(len(xi_real))
					for j in range(len(xi_real_)):
						xi_real_[j] = xi_real[j].real

					# Temperature gradient:
					nabla_star[i] = xi_real_**2 + K*xi_real_ + nabla_adiabatic[i]							 			  

					# Change in temperature with the mass:
					dT = (T[i]*nabla_star[i]/P[i])*dP 											 		   # dT/dm, convective

					# Flux:
					F_c[i]   = rho[i]*self.C_p*T[i]*sqrt(g*self.delta)*H_p**(-3./2)*(l_m/2)**2*xi_real_**3 # convective flux
					F_r[i]   = ((16*sigma*T[i]**4)/(3*kappa*rho[i]*H_p))*nabla_star[i]					   # radiative flux
					F_tot[i] = F_c[i] + F_r[i]															   # total flux
					

				# if not convectively unstable:
				else:
					nabla_star[i] = nabla_stable[i]
					#dT = -3.*kappa[0]*L[i]/(256*pi**2*self.sigma*r[i]**4*T[i]**3)			 			   # dT/dm, radiative
					dT = (T[i]*nabla_star[i]/P[i])*dP 
					
					# Flux:
					F_c[i]   = 0																		   # convective flux
					F_r[i]   = L[i]/(4*pi*r[i]**2)						  								   # radiative flux
					F_tot[i] = F_c[i] + F_r[i]															   # total flux
					

				# Test to implement dynamic step size:
				if dynamic == True:
					dm_r   = p*r[i]/dr
					dm_P   = p*P[i]/dP
					dm_L   = p*L[i]/dL
					dm_T   = p*T[i]/dT
					dm_all = array([dm_r,dm_P,dm_L,dm_T])
					#print 'dm_all=',dm_all

					if min(abs(dm_all)) < abs(dm1):
						dm = -min(abs(dm_all))
					else:
						dm = dm1

					m[i+1]   = m[i] + dm
					r[i+1]   = r[i] + dm*dr
					P[i+1]   = P[i] + dm*dP
					L[i+1]   = L[i] + dm*dL
					T[i+1]   = T[i] + dm*dT
					rho[i+1] = self.density(P[i+1], T[i+1])				# find rho from current value of P and T

				# constant step size:
				else:
					dm = dm1
					m[i+1]   = m[i] + dm
					r[i+1]   = r[i] + dm*dr
					P[i+1]   = P[i] + dm*dP
					L[i+1]   = L[i] + dm*dL
					T[i+1]   = T[i] + dm*dT
					rho[i+1] = self.density(P[i+1], T[i+1])

			tol  = 1e5													# find point where R0 is at 10%
			tol2 = 6e21													# find point where L0 is at 99.5%
			if abs(r[i]-0.1*r[0])<tol:
				print '0.1R0 is reached at index ', i, 'with L[i]/L0 = ', L[i]/L0
			if abs(L[i]-0.995*L[0])<tol2:
				print '0.995L0 is reached at index ', i, 'with r[i]/R0 = ', r[i]/R0

		'''
		#Plot luminosity, radius, temperature and density:
		plot(m[0:index]/M0, L[:index]/self.L_sun)
		xlabel(r'$m/M_0$', **axis_font)
		ylabel(r'$L/L_{\odot}$', **axis_font)
		title('Luminosity', **title_font)
		show()

		plot(m[0:index]/M0, r[:index]/self.R_sun)
		xlabel(r'$m/M_0$', **axis_font)
		ylabel(r'$r/R_{\odot}$', **axis_font)
		title('Radius', **title_font)		
		show()

		plot(m[0:index]/M0, T[:index]/1e6)
		xlabel(r'$m/M_0$', **axis_font)
		ylabel(r'$T\;[MK]$', **axis_font)
		title('Temperature', **title_font)
		show()
		
		plot(m[:index]/self.M0, rho[:index]/self.rho_sun_avg)
		xlabel(r'$m/M_0$', **axis_font)
		ylabel(r'$\rho/\overline{\rho}_{\odot}$', **axis_font)
		title('Density', **title_font)
		ylim(1,10)
		show()
		'''
		return m, r, P, L, T, rho, index, epsilon, F_c, F_r, F_tot, nabla_star, nabla_stable, nabla_adiabatic, PPI, PPII, E_tot


'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script calculating values of R0, T0 and rho0 giving r, L and m going to zero within 5% :
(This was used in project 1, but has not been used in this project, as gaining an accurate result
takes too much time)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
combo    = False
if combo == True:
	n         = 5														# number of values in array
	R0_frac   = linspace(0.4,1.2,n)										# chosen range of R0 values
	T0        = linspace(4000,7000,n)									# chosen range of T0 values
	rho0_frac = linspace(0.5*1.42*10**(-7),80*1.42*10**(-7),n)			# chosen range of rho0 values
	limit     = 1e4														# some random large limit

	#triple for-loop checking all combinations of R0, T0 and rho0: 
	for i in range(n):
		for j in range(n):
			for k in range(n):
				print i,j,k
				#get values:
				stellar_core_ = stellar_core(R0_frac[i], T0[j], rho0_frac[k])
				[m, r, P, L, T, rho, index, eps, Fc, Fr, Ftot, nabla_star, nabla_stable, nabla_adiabatic, PPI, PPII, Etot] = stellar_core_.ODE_solution(dynamic=True)
				m_lim   = m[index-1]/m[0]		  # last m value in fractions of M0
				r_lim   = r[index-1]/r[0]		  # last r value in fractions of R0
				L_lim   = L[index-1]/L[0]		  # last L value in fractions of L0
				lim_sum = m_lim+r_lim+L_lim 	  # sum of fractions - want this to be as low as possible
				if lim_sum < limit:
					limit  = lim_sum			  # lowest fraction-combination is new limit, and these values are printed
					print 'R0/R_sun = ',R0_frac[i], 'T0 = ',T0[j], 'rho0/rho_avg = ',rho0_frac[k]
					print 'r/R0 = ', r[index-1]/r[0], 'L/L0 = ',L[index-1]/L[0], 'm/M0 = ',m[index-1]/m[0]
					print '   ' 


'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script extracting the parameters describing the star for given initial values:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
extract = True
if extract == True:
	#stellar_core_ = stellar_core(1.2, 5770, 100*1.42*10**(-7))	# good initial values, convection in core
	stellar_core_  = stellar_core(1.2, 5500, 150*1.42*10**(-7))	# good initial values, no convection in core
	#stellar_core_ = stellar_core(1.0, 5770, 1.42*10**(-7))		# original initial values (the sun, except P0)
	[m, r, P, L, T, rho, index, eps, Fc, Fr, Ftot, nabla_star, nabla_stable, nabla_adiabatic, PPI, PPII, Etot] = stellar_core_.ODE_solution(dynamic=True)
	print 'initial values:','m =',m[0],'r =',r[0],'P =',P[0],'L =',L[0],'T =',T[0],'rho =',rho[0]
	print 'final values:', 'm =',m[index-1]/m[0],'M0','r =',r[index-1]/r[0],'R0','P =',P[index-1]/P[0],'P0'
	print 'final values:','L =',L[index-1]/L[0],'L0','T =',T[index-1]/T[0],'T0', 'rho =',rho[index-1]/rho[0],'rho0'
	

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Part of script plotting m, L, T, rho and P as functions of r:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''
plot_properties = True
if plot_properties == True:
	# plot m, L, T, rho, epsilon and P as functions of r:
	plot(r[:index],m[:index])
	xlabel(r'$r\;[m]$',**axis_font);ylabel(r'$m\;[kg]$',**axis_font);title('m(r) for best model',**title_font)
	show()
	plot(r[:index],L[:index])
	xlabel(r'$r\;[m]$',**axis_font);ylabel(r'$L\;[W]$',**axis_font);title('L(r) for best model',**title_font)
	show()
	plot(r[:index],T[:index])
	xlabel(r'$r\;[m]$',**axis_font);ylabel(r'$T\;[K]$',**axis_font);title('T(r) for best model',**title_font)
	show()
	semilogy(r[:index],rho[:index]/rho[0])
	hold('on')
	semilogy(r[:index],P[:index]/P[0])
	xlabel(r'$r\;[m]$',**axis_font)
	title(r'$P(r)$ and $\rho(r)$ for best model',**title_font)
	legend([r'$\log_{10}(\rho/\rho_0)$',r'$\log_{10}(P/P_0)$'])
	show()

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script plotting fraction of energy transported by convection and radition
as a function of radius:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''
radiation_frac = True
if radiation_frac == True:
	Fr_frac = Fr[:index]/Ftot[:index]
	Fc_frac = Fc[:index]/Ftot[:index]
	plot(r[:index]/r[0],Fr_frac)
	hold('on')
	plot(r[:index]/r[0],Fc_frac)
	legend([r'$F_r/(F_r+F_c)$',r'$F_c/(F_r+F_c)$'], loc='best')
	xlabel(r'$R/R_0$')
	ylabel(r'Fraction $[0,1]$')
	title('Energy transportion by convection/radiation')
	show()

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script plotting fraction of energy produced by PPI and PPII as a function 
of radius. The relative value of epsilon is also plotted:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''
energy_PP    = False
if energy_PP == True:
	eps_rel  = eps[:index]/max(eps[:index])
	PPI_rel  = PPI[:index]/Etot[:index]
	PPII_rel = PPII[:index]/Etot[:index]
	plot(r[:index],eps_rel)
	hold('on')
	plot(r[:index],PPI_rel)
	hold('on')
	plot(r[:index],PPII_rel)
	legend([r'$\varepsilon/\varepsilon_{max}$',r'$E_{PPI}/E_{tot}$', r'$E_{PPII}/E_{tot}$'], loc='best')
	xlabel(r'$R$')
	ylabel(r'Fraction $[0,1]$')
	title('Energy produced by PPI and PPII chains')
	show()


	#Parts of sun dominated by PPI and PPII:
	PPI_part = abs(trapz(PPI[:index]/Etot[:index], r[:index]/r[0]))
	PPII_part = abs(trapz(PPII[:index]/Etot[:index], r[:index]/r[0]))
	print 'Dominated by PPI:', PPI_part*100, '%'
	print 'Dominated by PPII:', PPII_part*100, '%'

	#Fraction of energy from PPI and PPII (where is most of the energy produced):
	PPI_frac  = trapz(PPI[:index])
	PPII_frac = trapz(PPII[:index])
	Etot_frac = trapz(Etot[:index])
	print 'Energy from PPI:', (PPI_frac/Etot_frac)*100, '%'
	print 'Energy from PPII:', (PPII_frac/Etot_frac)*100, '%'



'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script showing how width of convection zone changes with the initial density:
(higher initial density --> larger convection zone)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
change_density    = False
if change_density == True:
	rho0_frac     = linspace(1.42*10**(-7), 100*1.42*10**(-7),5)
	for l in range(len(rho0_frac)):
		stellar_core_ = stellar_core(1.0, 5770, rho0_frac[l])
		[m, r, P, L, T, rho, index, eps, Fc, Fr, Ftot, nabla_star, nabla_stable, nabla_adiabatic, PPI, PPII, Etot] = stellar_core_.ODE_solution(dynamic=True)
		plot(r[:index]/r[0],Fc[:index]/Ftot[:index], label=r'%.1e$\overline{\rho}_{\odot}$' % rho0_frac[l])
	legend(loc='best')	
	xlabel(r'$R/R_{\odot}$')
	ylabel(r'$F_c/(F_c+F_r)$')
	title('Fraction of flux coming from convection')
	show()


'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script showing how width of convection zone changes with the initial temperature:
(almost no difference when only changing T by a factor 5)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
change_temperature = False
if change_temperature == True:
	T0 = linspace(0.2*5770, 5.0*5770,5)
	for l in range(len(T0)):
		stellar_core_ = stellar_core(1.0, T0[l], 1.42*10**(-7))
		[m, r, P, L, T, rho, index, eps, Fc, Fr, Ftot, nabla_star, nabla_stable, nabla_adiabatic, PPI, PPII, Etot] = stellar_core_.ODE_solution(dynamic=True)
		plot(r[:index]/r[0],Fc[:index]/Ftot[:index], label='%.1e K' % T0[l])
	legend(loc='best')
	xlabel(r'$R/R_{\odot}$')
	ylabel(r'$F_c/(F_c+F_r)$')
	title('Fraction of flux coming from convection')	
	show()

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script showing how width of convection zone changes with the initial radius:
(higher initial radius --> larger convection zone)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''
change_radius    = False
if change_radius == True:
	R0 = linspace(1./5, 5.0,5)
	for l in range(len(R0)):
		stellar_core_ = stellar_core(R0[l], 5770, 1.42*10**(-7))
		[m, r, P, L, T, rho, index, eps, Fc, Fr, Ftot, nabla_star, nabla_stable, nabla_adiabatic, PPI, PPII, Etot] = stellar_core_.ODE_solution(dynamic=True)
		plot(r[:index]/r[0],Fc[:index]/Ftot[:index], label=r'$%.1f R_{\odot}$' % R0[l])
	legend(loc='best')
	xlabel(r'$R/R_0$')
	ylabel(r'$F_c/(F_c+F_r)$')
	title('Fraction of flux coming from convection')	
	show()


'''
- - - - - - - - - - - - - - - - - - - - - - - - - -
Part of script plotting the temperature gradients:
- - - - - - - - - - - - - - - - - - - - - - - - - -
'''
T_gradients    = False
if T_gradients == True:
	semilogy(r[:index]/r[0],nabla_stable[:index])
	hold('on')
	semilogy(r[:index]/r[0],nabla_star[:index])
	semilogy(r[:index]/r[0],nabla_adiabatic[:index])
	xlabel(r'$R/R_0$')
	legend([r'$\nabla_{stable}$',r'$\nabla^*$',r'$\nabla_{ad}$'], loc='best')
	title('Temperature gradients')
	show()


'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Part of script plotting a cross section of the star, giving us an overwiev of
the convection and radiation zones, and at the same time decide the widt of the
convection zone outside the core:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
'''
cross_section    = True
if cross_section == True:
	figure()
	fig = gcf()  # get current figure
	ax  = gca()  # get current axis
	#fixing axis:
	rmax = 1.2
	ax.set_xlim(-rmax,rmax)
	ax.set_ylim(-rmax,rmax)
	ax.set_aspect('equal')	# make the plot circular

	show_every      = 30
	core_limit      = 0.995*L[0]
	convection_zone = []		# convection outside the core, where does it start and end?
	j = show_every
	for k in range(0, index-1):
		j += 1
		if j >= show_every:
			# Outside the core:					
			if(L[k] > core_limit):
				# convection:			
				if(Fc[k] > 0.0):			
					circle_red = Circle((0,0),r[k]/r[0],color='red',fill=False)
					convection_zone.append(k)
					ax.add_artist(circle_red)
				# radiation:
				else:						
					circle_yellow = Circle((0,0),r[k]/r[0],color='yellow',fill=False)
					ax.add_artist(circle_yellow)
			# Inside the core:		
			else:
				# convection:					
				if(Fc[k] > 0.0):			
					circle_blue = Circle((0,0),r[k]/r[0],color='blue',fill = False)
					ax.add_artist(circle_blue)
				# radiation:
				else:						
					circle_cyan = Circle((0,0),r[k]/r[0],color='cyan',fill = False)
					ax.add_artist(circle_cyan)
			j = 0
	# create legends:
	circle_red    = Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)
	circle_yellow = Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
	circle_cyan   = Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
	circle_blue   = Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
	ax.legend([circle_red, circle_yellow, circle_cyan, circle_blue], ['Convection outside core', 'Radiation outside core', 'Radiation inside core', 'Convection inside core']) # only add one (the last) circle of each colour to legend
	legend(loc=2)
	xlabel(r'$R[R_0]$')
	ylabel(r'$R[R_0]$')
	title('Cross-section of star')
	# Show all plots
	show()

	#find width of convection zone outside core:
	start  = convection_zone[0]
	end    = convection_zone[-1]
	radius = (r[start]-r[end])/r[0]
	print 'The convection zone outside the core has size:', radius, 'R0'

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:32:43 2018

@author: william holtam
"""
#Import_Python_Libraries_______________________________________________________
import sys ## For checking the current version of python etc
import os ## For checking the current directory
import numpy as np ## For using array data structures
import pandas as pd ## For importing data as dataframe structures
import matplotlib.pyplot as plt ## For creating charts e.g. histograms
import matplotlib.lines as mlines ## For creating charts e.g. histograms
import matplotlib.patches as mpatches ## For creating chart legends
import matplotlib.ticker as mtick ## For altering tick marks on charts
#______________________________________________________________________________

#MIT Essay on Automation

#_Print_System_Information_____________________________________________________
cwd = os.getcwd()
print(cwd, "\n")

win = sys.getwindowsversion()
print(win, "\n")

python_ver = sys.version
print(python_ver, "\n")
#______________________________________________________________________________

#_Input_Starting_Population_and_Death_Rates____________________________________
data = pd.read_csv('data/Population.csv', delimiter=',', skiprows=2, 
                   header=None).values
m_pop = data[0,:]
f_pop = data[1,:]
m_prob_death = data[2,:]
f_prob_death = data[3,:]

children_born = m_pop[0] + f_pop[0] # Number of live births 2016
f_between_15_and_44 = sum(f_pop[15:45]) # Female population aged 15 to 44
# General Fertility Rate - Do not recalculate each year as it has plateaued.
GFR = children_born/f_between_15_and_44 
#______________________________________________________________________________

#_Import_Proportional_Change_in_Death_Rate_Per_Year____________________________
x_axis = range(114)
m_change_prob_death = data[6,:] # % Change in death rate for each age per year
m_std_err_death = data[7,:] #Standard Error in male death rate
f_change_prob_death = data[8,:] # % Change in death rate for each age per year
f_std_err_death = data[9,:] #Standard Error in male death rate

m_ymin = m_change_prob_death - m_std_err_death
m_ymax = m_change_prob_death + m_std_err_death

f_ymin = f_change_prob_death - f_std_err_death
f_ymax = f_change_prob_death + f_std_err_death

plt.plot(x_axis, m_change_prob_death, color='blue', label='male')
plt.plot(x_axis, f_change_prob_death, color='red', label='female')

plt.fill_between(x_axis, m_ymax, m_ymin, color='blue', alpha=0.3)
plt.fill_between(x_axis, f_ymax, f_ymin, color='red', alpha=0.3)

plt.grid(True)
plt.xlabel("Age")
plt.ylabel("Annual % Change in Death Rate")
plt.xlim([0,x_axis[-1]])
plt.ylim([-0.05,0.01])
plt.title("Annual Percentage Change in Death Rate Split by Gender")
plt.legend()
plt.show()
#______________________________________________________________________________

#_Create_a_List_of_Age_groups__________________________________________________
age_groups = []
for i in np.arange(0,99,5):
    start, end = 0+i, 4+i
    age_groups.append(str(start)+"-"+str(end))
age_groups.append("100+")
#______________________________________________________________________________

#_Import_NHS_Admissions_Data___________________________________________________
NHS_data = pd.read_csv('data/NHS Admissions1.csv', delimiter=',', skiprows=1, 
                       header=None).values

NHS_admissions = NHS_data[0]
NHS_net_expenditure = 120512e6
#https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/630570/60243_PESA_Accessible.pdf
#http://www.nhsconfed.org/resources/key-statistics-on-the-nhs

NHS_age_groups = ['0', '1-4', '5-9', '10-14', '15', '16', '17', '18', '19']
for i in np.arange(20,89,5):
    start, end = 0+i, 4+i
    NHS_age_groups.append(str(start)+"-"+str(end))
NHS_age_groups.append("90+")

total_pop_using_NHS_age_groups = []
total_data = (m_pop + f_pop).tolist()
total_pop_using_NHS_age_groups.append(total_data[0])
total_pop_using_NHS_age_groups.append(sum(total_data[1:5]))
total_pop_using_NHS_age_groups.append(sum(total_data[5:10]))
total_pop_using_NHS_age_groups.append(sum(total_data[10:15]))
for j in np.arange(15,20):
    total_pop_using_NHS_age_groups.append(total_data[j])
for i in np.arange(20,90,5):
    total_pop_using_NHS_age_groups.append(sum(total_data[(0+i):(5+i)]))
total_pop_using_NHS_age_groups.append(sum(total_data[90:]))

NHS_admis_prop_of_total_pop = (NHS_admissions/
                               np.array(total_pop_using_NHS_age_groups))

NHS_adm_of_pop = []
NHS_adm_of_pop.append(NHS_admis_prop_of_total_pop[0])
for i in np.arange(4):
    NHS_adm_of_pop.append(NHS_admis_prop_of_total_pop[1])
for i in np.arange(5):
    NHS_adm_of_pop.append(NHS_admis_prop_of_total_pop[2])
for i in np.arange(5):
    NHS_adm_of_pop.append(NHS_admis_prop_of_total_pop[3])
for j in np.arange(4,9):
    NHS_adm_of_pop.append(NHS_admis_prop_of_total_pop[j])
for j in np.arange(9,23):
    for i in np.arange(5):
        NHS_adm_of_pop.append(NHS_admis_prop_of_total_pop[j])
for i in np.arange(23,47):
    NHS_adm_of_pop.append(NHS_admis_prop_of_total_pop[23])

NHS_admissions_by_age = np.array(NHS_adm_of_pop)*total_data
NHS_cost_by_age = (NHS_admissions_by_age*NHS_net_expenditure/sum(NHS_admissions)).tolist()
NHS_cost_per_person = []
NHS_cost_per_person.append(NHS_cost_by_age)
total_NHS_cost = []
total_NHS_cost.append(sum(NHS_cost_per_person[0]))
#______________________________________________________________________________

#_Input_Immigration_Stats______________________________________________________
m_immigration_short = np.array([22,108,160,19,4])*1000
f_immigration_short = np.array([14,113,127,18,3])*1000

count = 0
m_migration_age_groups=[]
f_migration_age_groups=[]
for i in [m_migration_age_groups,f_migration_age_groups]:
    if count == 0:
        pop = m_pop
    else:
        pop = f_pop
    i.append(sum(pop[0:15]))
    i.append(sum(pop[15:25]))
    i.append(sum(pop[25:45]))
    if (count == 0):
        i.append(sum(pop[45:65]))
        i.append(sum(pop[65:]))
    else:
        i.append(sum(pop[45:60]))
        i.append(sum(pop[60:]))
    count += 1

m_immigration_proportion = m_immigration_short/np.array(m_migration_age_groups)
f_immigration_proportion = f_immigration_short/np.array(f_migration_age_groups)

count = 0
m_immigration = []
f_immigration = []
for j in [m_immigration,f_immigration]:
    if count ==0:
        immigration_proportion = m_immigration_proportion
        pop = m_pop
    else:
        immigration_proportion = f_immigration_proportion
        pop = f_pop
    for i in range(15):
        j.append(pop[i]*immigration_proportion[0])
    for i in range(15,25):
        j.append(pop[i]*immigration_proportion[1])
    for i in range(25,45):
        j.append(pop[i]*immigration_proportion[2])
    if count ==0:
        for i in range(45,65):
            j.append(pop[i]*immigration_proportion[3])
        for i in range(65,114):
            j.append(pop[i]*immigration_proportion[4])
    else:
        for i in range(45,60):
            j.append(pop[i]*immigration_proportion[3])
        for i in range(60,114):
            j.append(pop[i]*immigration_proportion[4])
    count += 1

m_emigration_short = np.array([-5,-45,-114,-20,-6])*1000
f_emigration_short = np.array([-9,-51,-71,-13,-6])*1000

m_emigration_proportion = m_emigration_short/np.array(m_migration_age_groups)
f_emigration_proportion = f_emigration_short/np.array(f_migration_age_groups)

count = 0
m_emigration = []
f_emigration = []
for j in [m_emigration, f_emigration]:
    if count == 0:
        emigration = m_emigration
        pop = m_pop
        emigration_proportion = m_emigration_proportion
    else:
        emigration = f_emigration
        pop = f_pop
        emigration_proportion = f_emigration_proportion        
    for i in range(15):
        j.append(pop[i]*emigration_proportion[0])
    for i in range(15,25):
        j.append(pop[i]*emigration_proportion[1])
    for i in range(25,45):
        j.append(pop[i]*emigration_proportion[2])
    if count == 0:
        for i in range(45,65):
            j.append(pop[i]*emigration_proportion[3])
        for i in range(65,114):
            j.append(pop[i]*emigration_proportion[4])
    else:
        for i in range(45,60):
            emigration.append(f_pop[i]*emigration_proportion[3])
        for i in range(60,114):
            emigration.append(pop[i]*emigration_proportion[4])
    count += 1
#______________________________________________________________________________

#_Create_Population_Pyramid_of_Initial_Population______________________________
f_trunk = []
m_trunk = []
for i in np.arange(0,99,5):
    start, end = 0+i, 4+i
    m_trunk.append(sum(m_pop[start:end]))
    f_trunk.append(sum(f_pop[start:end]))
m_trunk.append(sum(m_pop[100:]))
f_trunk.append(sum(f_pop[100:]))

pyramid_list = []
pyramid = []
for i in range(len(age_groups)):
    temp = []
    temp.append(age_groups[i])
    temp.append(m_trunk[i])
    temp.append(f_trunk[i])
    pyramid.append(temp)
pyramid_list.append(pyramid)
# Useful Notes:  https://github.com/afolaborn/Python_Jupyter_Notebook/blob/master/Population-Pyramid/Population_Pyramid_Final.ipynb

#_Model________________________________________________________________________
inflation = 0.02
x = 25 # Year of forecast to plot
m_timeseries = [m_pop]
f_timeseries = [f_pop]
for j in range(x):
    f_temp = [0]
    m_temp = [0]
    for i in np.arange(1,len(m_pop)+1):
        f_temp.append(f_pop[i-1])
        m_temp.append(m_pop[i-1])

    m_pop = m_temp
    f_pop = f_temp

    f_between_15_and_44 = sum(f_pop[15:45])

    sex_ratio = 1.053
    children_born = int(GFR*f_between_15_and_44)
    f_pop[0] = children_born*1/(1+sex_ratio)
    m_pop[0] = GFR*f_between_15_and_44*sex_ratio/(1+sex_ratio)
    del f_pop[114]
    del m_pop[114]

    # How much the death rate is expected to change per year for each age.
    m_prob_death = m_prob_death*(1+m_change_prob_death)
    f_prob_death = f_prob_death*(1+f_change_prob_death)

    m_pop = (np.array(m_pop)*(1-m_prob_death) + np.array(m_immigration) + 
              np.array(m_emigration)).tolist()
    f_pop = (np.array(f_pop)*(1-f_prob_death) + np.array(f_immigration) + 
              np.array(f_emigration)).tolist()

    m_timeseries.append(m_pop)
    f_timeseries.append(f_pop)

    f_trunk = []
    m_trunk = []
    for i in np.arange(0,99,5):
        start, end = 0+i, 4+i
        m_trunk.append(sum(m_pop[start:end]))
        f_trunk.append(sum(f_pop[start:end]))
    m_trunk.append(sum(m_pop[100:]))
    f_trunk.append(sum(f_pop[100:]))

    pyramid = []
    for i in range(len(age_groups)):
            temp = []
            temp.append(age_groups[i])
            temp.append(m_trunk[i])
            temp.append(f_trunk[i])
            pyramid.append(temp)
    pyramid_list.append(pyramid)

    total_pop = np.array(m_pop)+np.array(f_pop)

    NHS_admissions_by_age = np.array(NHS_adm_of_pop)*total_pop

    NHS_cost_by_age = (NHS_admissions_by_age*NHS_net_expenditure/
                               sum(NHS_admissions))#*(1+inflation))
    NHS_cost_by_age.tolist()
    NHS_cost_per_person.append(NHS_cost_by_age)

    total_NHS_cost.append(sum(NHS_cost_by_age))

timeseries = np.array(m_timeseries) + np.array(f_timeseries)

print("The General Fertility Rate is: ",GFR)
#______________________________________________________________________________

#_Tabulations__________________________________________________________________
x_axis = []
for i in range(x+1):
    x_axis.append(2016+i)
billions = lambda x,pos: '£%1.1fbn' % (x*1e-9)
formatter = mtick.FuncFormatter(billions)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.ylim(0.0,250e9)
plt.xlim(2016,2016+x)
plt.plot(x_axis, total_NHS_cost)
plt.title("Total NHS Costs in Real Terms")
plt.show()

billions = lambda x,pos: '£%1.1fbn' % (x*1e-9)
formatter = mtick.FuncFormatter(billions)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.xlim(0,113)
plt.ylim(0.0,10e9)
plt.plot(NHS_cost_per_person[0], color = 'blue', label = "2016")
plt.plot(NHS_cost_per_person[x], color = 'red', label = str(2016+x))
plt.xlabel('Age')
plt.ylabel("Cost")
plt.legend()
plt.title("NHS Cost Distribution by Age")
plt.show()

#_Initial_Population_2016
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=False)
plt.plot(m_timeseries[0], color='blue')
plt.plot(f_timeseries[0], color='red')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.title('UK Population by Individual Age in 2016')
blue_line = mlines.Line2D([], [], color='blue', label='Male')
red_line = mlines.Line2D([], [], color='red', label='Female')
plt.legend(handles=[blue_line, red_line])
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
plt.xlim([0,xmax])
plt.ylim([0,ymax])
plt.show()

year = str(2016+x) # Set Year of final tabulation as a string

#_Plot_Comparing_Initial_Population_and_Final_Population
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=False)
plt.plot(timeseries[0], color='blue')
plt.plot(timeseries[x], color='red')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.title('UK Population by Individual Age')
blue_line = mlines.Line2D([], [], color='blue', label='2016')
red_line = mlines.Line2D([], [], color='red', label=year)
plt.legend(handles=[blue_line, red_line])
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
plt.xlim([0,xmax])
plt.ylim([0,ymax])
plt.show()

#_Plot_Comparing_Initial_Population_and_Final_Population_Split_by_Gender
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=False)
plt.plot(m_timeseries[0], color='blue')
plt.plot(f_timeseries[0], color='red')
plt.plot(m_timeseries[x], color='blue', linestyle = 'dashed')
plt.plot(f_timeseries[x], color='red', linestyle = 'dashed')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.title('UK Population by Individual Age Split by Gender')
blue_line = mlines.Line2D([], [], color='blue', label='Male 2016')
red_line = mlines.Line2D([], [], color='red', label='Female 2016')
blue_dash_line = mlines.Line2D([], [], color='blue', linestyle = 'dashed',
                               label='Male '+year)
red_dash_line = mlines.Line2D([], [], color='red', linestyle = 'dashed',
                              label='Female '+year)
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
plt.xlim([0,xmax])
plt.ylim([0,ymax])
plt.legend(handles=[blue_line, red_line, blue_dash_line, red_dash_line])
plt.show()

#_UK_Population_Pyramid
for i in [0, int(x/4), int(x/2), int(3*x/4), int(x)]:
    year = str(2016+i)
    fig, axes = plt.subplots(ncols=2, sharey=True)

    pyramid_dataframe=pd.DataFrame(pyramid_list[i],
                                   columns=['Age Group', 'Male', 'Female'])

    pyramid_dataframe.plot.barh(x='Age Group', y='Male', color='blue',
                                label='Male', legend=True,ax=axes[0])
    pyramid_dataframe.plot.barh(x='Age Group', y='Female', color='red',
                                label='Female', legend=True, ax=axes[1])

    axes[0].set(yticklabels=[])
    axes[0].yaxis.tick_right()
    for i in range(2):
        axes[i].set_xlim([0,3e6])
    axes[0].invert_xaxis()
    y_axis = axes[0].get_yaxis()
    y_axis.set_label_text('foo')
    y_label = y_axis.get_label()
    y_label.set_visible(False)

    for i in range(len(age_groups)):
        axes[0].annotate(age_groups[i], (0.5, i), xycoords=('figure fraction',
            'data'), ha='center', va='center')

    for ax in axes.flat:
        ax.margins(0.03)

    fig.tight_layout()
    fig.subplots_adjust(top = 0.92,bottom = 0.14,wspace=0.25)
    fig.suptitle('UK Population Pyramid '+year)
    fig.text(0.5, 0.04, 'Population in Age Group', ha='center')
    axes[0].get_xaxis().set_major_formatter(
            mtick.FuncFormatter(lambda x, f_between_15_and_44: format('%1.1fM' % (x*1e-6))))
    axes[1].get_xaxis().set_major_formatter(
            mtick.FuncFormatter(lambda x, f_between_15_and_44: format('%1.1fM' % (x*1e-6))))

    axes[0].legend_.remove(), axes[1].legend_.remove()

    blue_patch = mpatches.Patch([], [], color='blue', label='Male')
    red_patch = mpatches.Patch([], [], color='red', label='Female')

    axes[1].legend(handles=[blue_patch, red_patch])

    plt.show()
#______________________________________________________________________________

#_Information_About_Change_in_Total_Size_of_UK_Population
print("The Size of the UK Population in 2016 is",
      "{:.2e}".format(sum(timeseries[0])))
print("The Size of the UK Population in", str(2016+x), "is",
      "{:.2e}".format(sum(timeseries[x])))
print("With a % change of", "{:.2%}".format((sum(timeseries[x])-
      sum(timeseries[0]))/sum(timeseries[0])), "\n")

m_mean = []
f_mean = []
count = 0
for gender in [m_timeseries,f_timeseries]:
    for k in range(0,x+1):
        age = 0
        temp_mean = 0
        for i in gender[k]:
            temp_mean += i*age
            age += 1
        if count == 0:
            m_mean.append(temp_mean/sum(gender[k]))
        else:
            f_mean.append(temp_mean/sum(gender[k]))
    count += 1

print("The mean male age of the UK Population in 2016 is", m_mean[0])
print("The mean female age of the UK Population in 2016 is", f_mean[1])
print("The mean male age of the UK Population in ", str(2016+x), "is",
      m_mean[x])
print("The mean female age of the UK Population in ", str(2016+x), "is",
      f_mean[x], "\n")
print("The Total NHS Expenditure in 2016 is: ", NHS_net_expenditure) ##
print("The Total NHS Expenditure in 2016 should be: ",
      sum(NHS_cost_per_person[0])) ## These two numbers should be the same but aren't!!!
#print("Proportional difference = ",
#      (sum(NHS_cost_per_person[0])-NHS_net_expenditure)/NHS_net_expenditure)
print("The Total NHS Expenditure in ", str(2016+x), "is",
      sum(NHS_cost_per_person[x]))
print("This is a change of ", "{:.2%}".format((sum(NHS_cost_per_person[x])-
      sum(NHS_cost_per_person[0]))/sum(NHS_cost_per_person[0])))
#______________________________________________________________________________
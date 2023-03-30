# [Linkpost] The global biomass of wild mammals

*This is a linkpost for https://www.pnas.org/doi/10.1073/pnas.2204892120*

## New Data

The following infographics are excerpted from the paper:

![Earth's mammals by total biomass](https://www.pnas.org/cms/10.1073/pnas.2204892120/asset/188a1505-95cb-4f42-a839-40d68d37f905/assets/images/large/pnas.2204892120fig04.jpg)

![Mammal Individual Counts and Biomass](https://www.pnas.org/cms/10.1073/pnas.2204892120/asset/71ac00b6-36f0-4d39-92ce-5cbf99845c30/assets/images/large/pnas.2204892120fig02.jpg)


## Background

The following Our World in Data image has been fairly popular:
![Life on Earth: the distribution of all global biomass](https://ourworldindata.org/uploads/2018/11/Global-Taxa-Biomass.png)
(Link to accompanying article: https://ourworldindata.org/life-on-earth )

This infographic was based on [The Biomass Distrbution on Earth](https://www.pnas.org/doi/10.1073/pnas.1711842115), which was published in 2018 by Y. M. Bar-On, R. Phillips, and R. Milo.

This paper attempted to give a rough estimate of the biomass of various taxa on Earth. The exact granularity of the taxon being measured varied based on the available data. For mammals, the authors separated the data into 3 classes: 
* Humans
* Livestock
* Wild mammals

The newly published paper improves on this by providing a more detailed breakdown of mammal biomass as well as individual counts.

## Numerical Comparison with Previous Paper
The old paper reported biomass in terms of dry carbon. The new paper uses wet mass, which includes both water and non-carbon content.

For consistency, we'll use the dry carbon values for both papers. The authors state that an $\frac{1}{6}$ is an appropriate conversion factor from wet mass to dry carbon.

Using this conversion factor, here's how much the overall estimates changed between the 2018 and 2023 papers:

| Taxon | 2018 (Megatons Carbon) | 2023 (Megatons Carbon) | Change |
| --- | --- | --- | --- |
| Human | 60 | 65 | +8.3% |
| Livestock | 100 | 105 | +5% |
| Wild Mammals | 7 | 10 | +42% |

## Discussion

### Bats
One surprising result is the number of bats. Looking at the source data [here](https://gitlab.com/milo-lab-public/mammal_biomass/-/blob/master/results/results_grouped_by_order.csv), we observe that there are about 55.802 billion wild bats. 
Overall, bats make up 2/3 of of wild mammal individuals. The next most popular taxon, Rodentia, only has 25.882 billion indviduals.

This has implications for wild animal welfare. While rodent welfare has been mentioned before, bat welfare is not very popular: I haven't found any resources relating to it so far.

Some research topics which are worth looking into:
* Which kind of bats make up most of the population?
    * Microbats or megabats?
* How much do bats suffer?
* Are there tractable ways to reduce bat suffering?

### Livestock
I was looking to see if they had individual counts for domesticated mammals as well, but I couldn't find it. According to [the appendix](https://www.pnas.org/doi/suppl/10.1073/pnas.2204892120/suppl_file/pnas.2204892120.sapp.pdf), the methodology for calculating the individual counts of livestock was to retrieve them from [the FAO](https://www.fao.org/faostat/en/#data/QCL).

I retreived the data for the year 2021, and plotted it:

Bar Plot:
![Livestock Individual Counts (Bar Plot)](https://github.com/pimpale/pimpale.github.io/raw/master/src/assets/mammalbiomass/individualcount.png)
(You can view the associated jupyter notebook [here](https://pimpale.github.io/assets/mammalbiomass/livestock.html))

I suspect that the bee count is probably an underestimate. The rest of the numbers seem reasonable though.
# Human-pose-recognition
	
  ****Zobrazovací kód****
  
  Hlavní main funkce načte parametry z příkazové řádky,viz tabulka
| Parametr  | Second Header | First Header  |
| ------------- | ------------- |------------- |
| input | Adresář pro vstupní soubory  |  . (aktuální adresář) |
| output  | Adresář pro výstupní soubory  | .(aktuální adresář)  | 
| show  | Zobrazovat výstupní obrázky interaktivně   | True | 

Vytvoří se instance dvou základních tásků TaskMask a TaskKp. Následně se prochází adresář vstupních souborů, kde se vybírají pouze obrázky jpg, png, jpeg a dále se odfiltrují již segmentované výstupní soubory obsahující postfix _mask nebo _kp. Každý tásk obrázek načte, zpracuje a uloží do výstupního adresáře případně ihned po zpracování zobrazí uživateli. Pokud TaskMask nalezne aspoň jednu osobu spouští se také vyhledání keypoints následně vyhodnocuje i bodyparts. 

***TaskMask***  
Vstupní obrázek se převede na numpy array poté na tenzor, provede se klasifikace prostřednictvím modelu maskrcnn_resnet50_fpn a výpočet masek, zpracuje se pravděpodobnostní rozdělení klasifikovaných objektů, vyberou se objekty s pravděpodobností min 0.5. Dále se prochází jednotlivé masky a zajistí se jejich grafické zvýraznění modifikací barvy bodu. Výstupní obrázek obsahující masky nalezených objektů a nejmenší obepínající obdelníky se podle nastavení zobrazí uživateli a vždy se uloží na disk. 

***TaskKp***   
Vstupní obrázek se převede na numpy array poté na tenzor, takto vytvořený vstupní vektor vstupuje do modelu keypointrcnn_resnet50_fpn, který vrací nalezené objekty typu (score,keypoints). Score má význam pravděpodobnosti, že se jedná o osobu a vybírají se pouze objekty s pravděpodobností > 0.9. Pro takto vybrané objekty se do původního obrázku zakreslí keypointy. Propojení keypointů je staticky definované, prediktor vrací pouze souřadnice bodů v definovaném pořadí. Výsledný obrázek, se volitelně zobrazí při zpracování a současně se vždy uloží do nastaveného výstupního adresáře. 

***BodyParts***   
Modifikací výstupu neuronové sítě v podobě keypointů bylo dosaženo schopnosti rozpoznání jednotlivých částí končetin lidského těla. Tato modifikace je založena na výpočtu konstanty za pomoci vzdálenosti určitých keypointů a rozšíření spojnice mezi nimi, popřípadě vytvoření určitého geometrického tvaru na správné pozice.
Předtrénovaná neuronová síť ResNet-50 na základě vstupního obrázku vrací 3 sady souřadnic keypointů, které by měly být umístěny v klíčových bodech definujících lidské tělo. Každá sada je zároveň ohodnocena tzv. keypoint score, což nám říká, jak přesně jsou keypointy umístěny vzhledem k "vědomostem", které neuronový model při tréninku získal. Algoritmus dále využívá tu nejlepší sadu, která musí zároveň splňovat keypoint score větší, než 0.9.
 Máme-li tedy vhodnou sadu souřadnic 17 keypointů, můžeme pokročit k metodám, které vhodně rozšíří spojnice těchto bodů na vhodném místě a označí tak jednotlivé lidské končetiny.
 Šířka spojnice keypointů rukou je definována vektorovou délkou začátek - konec předloktí  dále vynásobenou konstantou viz. optimizer, která je vypozorována testováním na více obrázcích
 Šířka spojnice nohou je řešena stejným algoritmem, pouze měříme vektorovou vzdálenost bodů začátek - konec stehna a násobíme ho konstantou viz. optimizer.
 Označení oblasti hlavy vychází ze spojnice uší. Nalezneme střed spojnice uší, ve středu vztyčíme kolmici a na kolmici vyznačíme dva body symetricky kolem středu, jejichž vektorovou vzdálenost určíme pomocí vzdálenosti uší a konstanty viz. optimizer. Mezi těmito body vykreslíme spojnici o tloušťce  vektorové vzdálenosti uší.
Trup je rozeznáván pomocí 4 keypointů, 2 v ramenou a 2 na začátku nohou. Naleznou se středy těchto dvojic a vektorová vzdálenost se symetricky zmenší na hodnotu viz optimizer. Poté se vykreslí spojnice mezi těmito středy s tloušťkou, která se rovná vektorové vzdálenosti bodů v ramenou.

***Optimizer***   
Algoritmus zvýraznění body parts je založen na rozšíření spojnic definovaných keypointů, které jsou výstupem z natrénované neuronové site resnet_50_keypoints. Keypointy dávají lokaci a rošíření poskytuje výplň definovaných  oblastí jednotlivých částí těla.  Rozšíření je navrženo takovým způsobem, že šířka výplně je odvozena z délky příslušných spojnic bodů. Jedná se o lineární funkční závislost, kdy vzniká potřeba vhodným způsobem určit multiplikativní konstanty v definici výplně. Pro určení se využívá optimalizační strategie, kdy definujeme hodnotící funkci jako rozdíl počtu bodů které vyhovují ground true a počty bodů, které jsou mimo oblast ground true.  Algoritmus je vyhodnocován přes definovaný interval multiplikativnich konstant. Výsledkem jsou jednotlivé závislosti hodnotícího kriteria na hodnotě konstanty v daném intervalu pro každou body part. Jednotlié funkce vykazují hladký průběh s výrazným maximum, kde  konstantu určíme právě z tohoto extremu

***Intersection Over Union (IoU)***  
Vyhodnocení úspěšnosti mé nadstavby na keypoint R-CNN pro rozlišování lidských končetin, bylo řešeno metodou porovnávání pixelů na groud truth obrázku, tedy dokonalé segmentovaného lidského těla, s pixely na obrázku vyhodnoceném právě řešeným algoritmem.
Testovací sada byla vybrána z volně dostupného datasetu MPII. Jedná se o dataset zaměřený na lidské postavy v různých kontextech s různými světelnými podmínkam  

Vyhodnocovací script byl napsán v jazyce python. Pomocí knihovny numpy zpracovává vstupní obrázky ve formě rgb tezorů a počítá počet například červené (255,0,0) pixely, kterým přiřadí význam dolní část ruky. Tento postup je aplikován na obrázek s groud truth i na obrázek z rozpoznávacího algoritmu.
„Score“ na obrázku vyhodnoceném algoritmem je však podmíněno tak, že je přičten bod pouze v momentě, kdy je stejný pixel vyhodnocen jako např. dolní část ruky, na obou obrázcích. Nebo-li je červený (255,0,0) i na referenčím obrázku.
Poměrem těchto získaných počtů barevných pixelů získáváme hodnotu IoU (Intersection over Union).  Hodnoty IoU jsou vyhodnocovány zvlášť pro každou končetinu a dále průměrovány s postupem zpracování celého testovacího subsetu. Vyhodnocení lze vidět v grafu.




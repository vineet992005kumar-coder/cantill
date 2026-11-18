"""Small built-in procurement concept lexicon (no category names of any client). Used only in the zero-label path:
item text and tree-path text are both expanded with shared concept tokens so 'courier' can meet 'Freight' and
'toner' can meet 'Office Supplies'. Terms are word prefixes (>=4 chars) or exact short words; a few common
Spanish/Portuguese/German/French/Italian words are included."""
import re

CONCEPTS = {
 "software": "software licen subscri saas cloud hosting app lizenz licencia licença logiciel",
 "itequip": "laptop computer desktop monitor server printer scanner network router switch firewall storage notebook ordenador",
 "telecom": "telecom telephon mobile phone sim internet broadband bandwidth voip telefon teléfono celular",
 "itservice": "helpdesk cyber security it-service integration support-contract managed-service backup",
 "office": "stationery paper toner cartridge pen folder envelope office furniture desk chair papel papier bürobedarf",
 "printing": "printing print brochure leaflet flyer catalog signage banner label-print",
 "consulting": "consult advisory advisor strategy management-service beratung asesor consultoría consultoria",
 "legal": "legal lawyer attorney solicitor notary law patent trademark abogado anwalt juridique",
 "accounting": "audit accounting bookkeep tax-advice payroll-service accountant treuhand contabil",
 "recruit": "recruit staffing headhunt agency temp temporary personnel manpower labour labor contractor zeitarbeit",
 "training": "training course seminar workshop education tuition conference coaching formación capacitação schulung",
 "marketing": "marketing advertis campaign media promotion sponsor branding agency-fee seo social-media exhibition trade-show publicidad werbung",
 "research": "research survey study analysis laboratory testing certification inspection calibration",
 "insurance": "insurance premium policy liability coverage broker underwriting seguro versicherung assurance",
 "banking": "bank-fee bank charge interest loan financ commission guarantee fx-fee",
 "cleaning": "cleaning janitor sanit detergent disinfect hygiene mop waste-bin limpieza reinigung nettoyage pulizia",
 "waste": "waste recycl disposal scrap skip refuse hazardous-waste residuos abfall",
 "security": "security guard alarm cctv surveillance access-control patrol vigilancia wachdienst",
 "pest": "pest rodent fumigat extermin",
 "landscape": "landscap garden grounds lawn tree-cutting",
 "maintenance": "maintenance repair servicing overhaul refurb mantenimiento manutenção wartung maintenance reparación reparo reparatur",
 "hvac": "hvac air-condition aircon chiller boiler heating ventilation refrigerat compressor cooling",
 "electrical": "electric cable wire switchgear transformer breaker fuse relay contactor plc sensor lamp led lighting luminaire busbar substation",
 "plumbing": "plumb pipe pipework fitting valve tap drain sewer sanitary",
 "building": "building construction civil concrete cement brick scaffold roofing paint flooring carpentry joinery plaster steelwork",
 "rent": "rent lease tenancy premises warehouse-rent alquiler miete loyer",
 "utilities": "electricity power-supply gas-supply water-supply utility energy kwh megawatt tariff strom electricidad",
 "fuel": "fuel diesel petrol gasoline lubricant fuel-card bunker propane lpg kerosene combustible kraftstoff",
 "freight": "freight shipping courier postage post parcel haulage forwarding cargo express delivery transport logistic fracht flete frete envio versand",
 "customs": "customs duty duties brokerage clearance tariff import export aduana zoll",
 "warehouse": "warehous storage distribution fulfil 3pl handling-fee depot",
 "vehicle": "vehicle truck lorry van car fleet forklift tractor trailer tyre tire wheel chassis automotive",
 "travel": "travel flight airline airfare hotel accommodation lodging taxi car-rental mileage visa trip viaje reise voyage",
 "catering": "catering restaurant meal food beverage coffee canteen lunch snack vending water-cooler comida essen",
 "uniform": "uniform workwear clothing apparel garment footwear boots overall laundry textile",
 "ppe": "ppe glove helmet goggles respirator mask earplug safety-vest harness hi-vis hard-hat fire-extinguisher first-aid",
 "tools": "tool drill cutter blade abrasive grinding wrench hammer saw chisel toolbox insert-tool herramienta werkzeug outil",
 "fastener": "bolt nut screw washer rivet fastener anchor stud pin clip",
 "bearing": "bearing gearbox gear sprocket coupling shaft bushing roller chain belt pulley",
 "pump": "pump valve actuator hydraulic pneumatic cylinder hose fitting-hose manifold",
 "motor": "motor engine generator alternator drive inverter turbine",
 "filter": "filter cartridge-filter element strainer separator",
 "lubricant": "lubric grease oil coolant hydraulic-oil gear-oil",
 "gasket": "gasket seal o-ring packing sealant",
 "weld": "weld electrode brazing solder flux",
 "sparepart": "spare part component assembly kit replacement oem repuesto ersatzteil pièce peça",
 "chemical": "chemical solvent acid alkali caustic reagent catalyst additive compound chemikalie quimico",
 "adhesive": "adhesive glue sealant epoxy tape",
 "coating": "coating paint varnish ink lacquer primer pigment",
 "plastic": "plastic polymer resin polyethylene polypropylene pvc pet pellet granulate injection-mould kunststoff plástico",
 "rubber": "rubber elastomer silicone foam",
 "steel": "steel iron alloy stainless tinplate galvanized rebar beam sheet-metal coil-steel stahl acero aço",
 "aluminium": "alumin ingot can-stock extrusion alloy-al",
 "copper": "copper brass bronze zinc nickel tin lead metal-premium",
 "wood": "wood timber lumber plywood pallet-wood sawn",
 "paperboard": "cardboard carton corrugat paperboard kraft pulp box-board",
 "packaging": "packag carton box bag film wrap strapping stretch bottle jar closure lid cap crate container-packaging pallet embalaje verpackung emballage",
 "glass": "glass glazing mirror",
 "stone": "stone aggregate gravel sand ballast quarry crush dust",
 "explosive": "explosive detonator blasting powder emulsion",
 "mining": "mining miner roof-bolt drill-rod drill-bit ore shaft underground continuous-miner conveyor-belt",
 "medical": "medical surgical clinical syringe needle catheter bandage dressing implant diagnostic imaging x-ray hospital patient",
 "pharma": "pharma drug medicine tablet vaccine antibiotic reagent-lab prescription",
 "lab": "laboratory lab consumable pipette sample assay test-kit",
 "food": "ingredient flour sugar milk dairy meat vegetable fruit oil-food grain seed flavour fragrance",
 "agri": "fertili pesticide herbicide seed feed crop irrigation farm tractor-agri",
 "textile": "fabric yarn cotton thread fibre fiber leather textile-raw",
 "furniture": "furniture shelving cabinet locker mattress bed",
 "hr": "benefit pension medical-aid wellness recruitment-fee relocation allowance",
 "membership": "membership subscription-fee association dues levy fee",
 "donation": "donation charity sponsorship grant",
 "equipment": "equipment machine machinery plant apparatus instrument",
 "crane": "crane hoist lift elevator escalator winch",
 "ventilation": "ventilation fan duct blower extractor",
 "water": "water treatment filtration osmosis pump-water pipe-water",
}

_PAT = {}
for _c, _t in CONCEPTS.items():
    _PAT[_c] = [w for w in _t.replace("-", " ").split() if w]


def concept_tokens(text):
    """Return a string of 'cpt_<name>' tokens for every concept hit in the text."""
    words = re.findall(r"[a-zà-ÿ0-9]+", str(text).lower())
    if not words:
        return ""
    hits = []
    for name, stems in _PAT.items():
        for w in words:
            if len(w) < 3:
                continue
            if any((w.startswith(s) if len(s) >= 4 else w == s) for s in stems):
                hits.append("cpt_" + name)
                break
    return " ".join(hits)

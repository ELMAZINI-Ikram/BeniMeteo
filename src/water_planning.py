"""
water_planning.py — Planification Réservoirs d'Eau — BeniMeteo V4
"""
import math, io
import pandas as pd
import numpy as np
from datetime import datetime

EAU_POTABLE_L_PERS_J     = 5.0
EAU_AUTRES_L_PERS_J      = 35.0
EAU_TOTAL_L_PERS_J       = 40.0
FACTEUR_STRESS_THERMIQUE  = 1.25
SEUIL_STRESS_THERMIQUE    = 40.0
FACTEUR_ISOLEMENT         = 2.0
SEUIL_ISOLEMENT_KM        = 60.0
FACTEUR_SOURCE_NATURELLE  = 0.50
VOLUME_RESERVOIR_M3       = 5.0
COUT_RESERVOIR_5M3        = 8_000
COUT_INSTALLATION         = 2_000
COUT_TRANSPORT_PAR_KM     = 50

HOPITAUX_REGION = [
    {"nom":"CHR Béni Mellal",        "lat":32.338,"lon":-6.361},
    {"nom":"Hôpital Azilal",         "lat":31.965,"lon":-6.567},
    {"nom":"Hôpital Fqih Ben Saleh", "lat":32.502,"lon":-6.690},
    {"nom":"CHR Khénifra",           "lat":32.934,"lon":-5.672},
    {"nom":"CHR Khouribga",          "lat":32.881,"lon":-6.909},
    {"nom":"Hôpital Kasba Tadla",    "lat":32.601,"lon":-6.271},
    {"nom":"Centre Santé Demnate",   "lat":31.734,"lon":-7.005},
    {"nom":"Hôpital Midelt",         "lat":32.685,"lon":-4.734},
]

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0
    dl=math.radians(lat2-lat1); dl2=math.radians(lon2-lon1)
    a=math.sin(dl/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dl2/2)**2
    return R*2*math.asin(math.sqrt(a))

def distance_centre_sante(lat,lon,hopitaux=None):
    hop=hopitaux or HOPITAUX_REGION
    return round(min(haversine_km(lat,lon,h["lat"],h["lon"]) for h in hop),1)

def calculer_douar(row, tmax=25.0, source_naturelle=False, hopitaux=None):
    pop=int(row.get("population",0))
    menages=int(row.get("nb_menages",max(1,pop//5)))
    if pop==0: return {}
    lat=float(row.get("lat",32.337)); lon=float(row.get("lon",-6.349))
    alt=float(row.get("altitude",400))
    pct_m15=float(row.get("pct_moins_15",30))
    pct_60p=float(row.get("pct_60_plus",12))
    pct_vuln=float(row.get("pct_vulnerable",40))
    dist_km=distance_centre_sante(lat,lon,hopitaux)
    isole=dist_km>SEUIL_ISOLEMENT_KM

    besoin_potable=pop*EAU_POTABLE_L_PERS_J
    besoin_autres=pop*EAU_AUTRES_L_PERS_J
    stress=(tmax>SEUIL_STRESS_THERMIQUE)
    if stress:
        besoin_potable*=FACTEUR_STRESS_THERMIQUE
        besoin_autres*=FACTEUR_STRESS_THERMIQUE
    if source_naturelle:
        besoin_autres*=FACTEUR_SOURCE_NATURELLE
    besoin_total=besoin_potable+besoin_autres
    jours=2.0 if isole else 1.0
    volume_m3=round((besoin_total*jours)/1000,2)
    nb_res=math.ceil(volume_m3/VOLUME_RESERVOIR_M3)
    cout=nb_res*(COUT_RESERVOIR_5M3+COUT_INSTALLATION)+dist_km*COUT_TRANSPORT_PAR_KM

    s_vuln=min(pct_vuln/100,1.0)*40
    s_isol=min(dist_km/120,1.0)*30
    s_therm=20 if stress else 0
    s_alt=min(alt/3000,1.0)*10
    score=round(s_vuln+s_isol+s_therm+s_alt,1)

    if score>=60:    niv,col="CRITIQUE","#DC2626"
    elif score>=40:  niv,col="HAUTE","#EA580C"
    elif score>=20:  niv,col="MOYENNE","#CA8A04"
    else:            niv,col="NORMALE","#16A34A"

    return {
        "province":row.get("province_fr",""), "cercle":row.get("cercle",""),
        "commune":row.get("commune",""), "fraction":row.get("fraction",""),
        "douar":row.get("douar",""), "lat":lat,"lon":lon,"altitude_m":int(alt),
        "population":pop,"nb_menages":menages,
        "pct_moins_15":round(pct_m15,1),"pct_60_plus":round(pct_60p,1),
        "pct_vulnerable":round(pct_vuln,1),
        "dist_centre_sante_km":dist_km,"isole":isole,
        "tmax":tmax,"stress_thermique":stress,"source_naturelle":source_naturelle,
        "besoin_potable_L_j":round(besoin_potable),
        "besoin_autres_L_j":round(besoin_autres),
        "besoin_total_L_j":round(besoin_total),
        "L_par_pers_j":round(besoin_total/pop,1),
        "jours_autonomie":jours,"volume_total_m3":volume_m3,
        "nb_reservoirs_5m3":nb_res,"cout_estime_dh":round(cout),
        "score_priorite":score,"niveau_priorite":niv,"couleur_priorite":col,
    }

def calculer_region(hcp_df, tmax=25.0, source_naturelle_global=False, hopitaux=None):
    rows=[]
    for _,row in hcp_df.iterrows():
        r=calculer_douar(row.to_dict(),tmax,source_naturelle_global,hopitaux)
        if r: rows.append(r)
    df=pd.DataFrame(rows)
    if len(df)>0:
        df=df.sort_values("score_priorite",ascending=False).reset_index(drop=True)
        df.insert(0,"rang",range(1,len(df)+1))
    return df

def stats_region(df):
    if len(df)==0: return {}
    return {
        "nb_douars":int(len(df)),
        "population_totale":int(df["population"].sum()),
        "eau_potable_m3_j":round(df["besoin_potable_L_j"].sum()/1000,1),
        "eau_autres_m3_j":round(df["besoin_autres_L_j"].sum()/1000,1),
        "eau_totale_m3_j":round(df["besoin_total_L_j"].sum()/1000,1),
        "nb_reservoirs_total":int(df["nb_reservoirs_5m3"].sum()),
        "cout_total_dh":int(df["cout_estime_dh"].sum()),
        "nb_critique":int((df["niveau_priorite"]=="CRITIQUE").sum()),
        "nb_haute":int((df["niveau_priorite"]=="HAUTE").sum()),
        "nb_moyenne":int((df["niveau_priorite"]=="MOYENNE").sum()),
        "nb_normale":int((df["niveau_priorite"]=="NORMALE").sum()),
        "nb_isoles":int(df["isole"].sum()),
        "vol_total_m3":round(df["volume_total_m3"].sum(),1),
    }

def generer_csv(df):
    cols=["rang","niveau_priorite","province","cercle","commune","fraction","douar",
          "population","nb_menages","pct_moins_15","pct_60_plus","pct_vulnerable",
          "dist_centre_sante_km","isole","altitude_m","tmax","stress_thermique",
          "source_naturelle","besoin_potable_L_j","besoin_autres_L_j",
          "besoin_total_L_j","L_par_pers_j","jours_autonomie","volume_total_m3",
          "nb_reservoirs_5m3","cout_estime_dh","score_priorite","lat","lon"]
    cols=[c for c in cols if c in df.columns]
    return df[cols].to_csv(sep=";",index=False,encoding="utf-8-sig").encode("utf-8-sig")

def generer_pdf(df, stats, tmax=25.0, source_naturelle=False):
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        return b""

    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=landscape(A4),
        leftMargin=1.2*cm,rightMargin=1.2*cm,topMargin=1.2*cm,bottomMargin=1.2*cm)

    BL=colors.HexColor("#1A3D6B"); BCL=colors.HexColor("#EFF6FF")
    RG=colors.HexColor("#DC2626"); OR=colors.HexColor("#EA580C")
    JN=colors.HexColor("#CA8A04"); VT=colors.HexColor("#16A34A")
    GH=colors.HexColor("#F3F4F6"); GB=colors.HexColor("#E5E7EB")
    WH=colors.white; NK=colors.HexColor("#111827")

    styles=getSampleStyleSheet()
    def st(name,**kw):
        return ParagraphStyle(name,parent=styles["Normal"],**kw)

    today=datetime.now().strftime("%d/%m/%Y à %H:%M")

    # Header
    hdr=Table([[Paragraph(
        f"<font color='white'><b>BeniMeteo — Plan d'Intervention Eau — Région Béni Mellal-Khénifra</b><br/>"
        f"<font size='9'>{today} · Tmax={tmax}°C · {'Source naturelle prise en compte' if source_naturelle else 'Sans source naturelle'}</font></font>",
        st("H",fontSize=14,alignment=TA_CENTER,fontName="Helvetica-Bold")
    )]], colWidths=[26.6*cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),BL),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
    ]))

    # Métriques
    def mc(val,lab,bg=GH,tc=BL):
        return [Paragraph(f'<b>{val}</b>',st("V",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold",textColor=tc)),
                Paragraph(lab,st("L",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280")))]
    s=stats
    mdata=[
        [mc(f"{s.get('nb_douars',0):,}","Douars"),
         mc(f"{s.get('eau_totale_m3_j',0):.0f} m³","Eau/jour"),
         mc(f"{s.get('nb_reservoirs_total',0):,}","Réservoirs 5m³"),
         mc(f"{s.get('cout_total_dh',0)/1e6:.1f}M DH","Coût total"),
         mc(f"{s.get('nb_critique',0):,}","CRITIQUES",bg=colors.HexColor("#FEE2E2"),tc=RG),
         mc(f"{s.get('nb_haute',0):,}","HAUTES",bg=colors.HexColor("#FEF3C7"),tc=OR),
         mc(f"{s.get('nb_isoles',0):,}","Isolés >60km")],
    ]
    mdata_flat=[[c for pair in mdata[0] for c in pair]]
    mt=Table([[Paragraph(f'<b>{s.get("nb_douars",0):,}</b>',st("x",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold")),
               Paragraph(f'<b>{s.get("eau_totale_m3_j",0):.0f} m³</b>',st("x",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold")),
               Paragraph(f'<b>{s.get("nb_reservoirs_total",0):,}</b>',st("x",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold")),
               Paragraph(f'<b>{s.get("cout_total_dh",0)/1e6:.1f}M DH</b>',st("x",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold")),
               Paragraph(f'<b>{s.get("nb_critique",0):,}</b>',st("x",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold",textColor=RG)),
               Paragraph(f'<b>{s.get("nb_haute",0):,}</b>',st("x",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold",textColor=OR)),
               Paragraph(f'<b>{s.get("nb_isoles",0):,}</b>',st("x",fontSize=13,alignment=TA_CENTER,fontName="Helvetica-Bold")),],
              [Paragraph("Douars",st("y",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280"))),
               Paragraph("Eau/jour",st("y",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280"))),
               Paragraph("Réservoirs 5m³",st("y",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280"))),
               Paragraph("Coût total estimé",st("y",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280"))),
               Paragraph("CRITIQUES",st("y",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280"))),
               Paragraph("HAUTES",st("y",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280"))),
               Paragraph("Isolés >60km",st("y",fontSize=8,alignment=TA_CENTER,textColor=colors.HexColor("#6B7280"))),]],
             colWidths=[3.8*cm]*7)
    mt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),GH),("BACKGROUND",(4,0),(4,-1),colors.HexColor("#FEE2E2")),
        ("BACKGROUND",(5,0),(5,-1),colors.HexColor("#FEF3C7")),
        ("GRID",(0,0),(-1,-1),0.5,GB),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))

    # Tableau principal
    top=df.head(50)
    hdrs=["Rang","Priorité","Province","Commune","Douar","Pop.","% Vuln",
          "Dist CS\n(km)","Isolé","Potable\nL/j","Autres\n L/j","Total\nL/j",
          "L/pers","Vol.\nm³","Rés.\n5m³","Coût DH","Score"]
    cw=[0.8,1.5,2.2,2.3,2.8,0.9,1.0,1.1,0.9,1.2,1.2,1.2,0.9,1.0,0.9,1.6,0.9]

    tdata=[hdrs]
    for _,r in top.iterrows():
        tdata.append([
            str(int(r.get("rang",0))),r.get("niveau_priorite",""),
            str(r.get("province",""))[:14],str(r.get("commune",""))[:16],
            str(r.get("douar",""))[:18],f'{int(r.get("population",0)):,}',
            f'{r.get("pct_vulnerable",0):.0f}%',
            f'{r.get("dist_centre_sante_km",0):.0f}',
            "OUI" if r.get("isole") else "—",
            f'{int(r.get("besoin_potable_L_j",0)):,}',
            f'{int(r.get("besoin_autres_L_j",0)):,}',
            f'{int(r.get("besoin_total_L_j",0)):,}',
            f'{r.get("L_par_pers_j",40):.0f}',
            f'{r.get("volume_total_m3",0):.1f}',
            str(int(r.get("nb_reservoirs_5m3",0))),
            f'{int(r.get("cout_estime_dh",0)):,}',
            f'{r.get("score_priorite",0):.0f}',
        ])

    main=Table(tdata,colWidths=[w*cm for w in cw],repeatRows=1)
    ts=[
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),
        ("BACKGROUND",(0,0),(-1,0),BL),("TEXTCOLOR",(0,0),(-1,0),WH),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("GRID",(0,0),(-1,-1),0.3,GB),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WH,GH]),
    ]
    ncolors={"CRITIQUE":(colors.HexColor("#FEE2E2"),RG),
             "HAUTE":(colors.HexColor("#FEF3C7"),OR),
             "MOYENNE":(colors.HexColor("#FFF9C4"),JN),
             "NORMALE":(colors.HexColor("#DCFCE7"),VT)}
    for i,(_,r) in enumerate(top.iterrows(),1):
        n=r.get("niveau_priorite","NORMALE")
        if n in ncolors:
            bg,fg=ncolors[n]
            ts.extend([("BACKGROUND",(1,i),(1,i),bg),("TEXTCOLOR",(1,i),(1,i),fg),
                        ("FONTNAME",(1,i),(1,i),"Helvetica-Bold")])
        if r.get("isole"):
            ts.extend([("BACKGROUND",(8,i),(8,i),colors.HexColor("#FEF3C7")),
                        ("TEXTCOLOR",(8,i),(8,i),OR),("FONTNAME",(8,i),(8,i),"Helvetica-Bold")])
    main.setStyle(TableStyle(ts))

    # Note
    note=Table([[
        Paragraph("Base : 5 L/pers/j (eau potable) + 35 L/pers/j (usage) = 40 L/pers/j. "
                  f"Stress thermique (>{SEUIL_STRESS_THERMIQUE}°C) : ×{FACTEUR_STRESS_THERMIQUE}. "
                  f"Isolement (>{SEUIL_ISOLEMENT_KM}km) : autonomie 48h. "
                  f"Source naturelle : −50% hors-boisson. Réservoirs de {VOLUME_RESERVOIR_M3}m³. "
                  "Sources : OMS (2011) · Sphere (2018) · ONEE Maroc · RGPH 2024 HCP.",
                  st("NL",fontSize=7.5,textColor=colors.HexColor("#6B7280"))),
        Paragraph("BeniMeteo · USMS ENSA Béni Mellal",
                  st("NR",fontSize=7.5,alignment=TA_RIGHT,textColor=colors.HexColor("#9CA3AF"))),
    ]],colWidths=[19*cm,7.6*cm])
    note.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),4),
                               ("LINEABOVE",(0,0),(-1,0),0.5,GB),
                               ("VALIGN",(0,0),(-1,-1),"TOP")]))

    doc.build([hdr,Spacer(1,0.25*cm),mt,Spacer(1,0.25*cm),
               Paragraph("Top 50 douars prioritaires",
                          st("SEC",fontSize=10,textColor=BL,fontName="Helvetica-Bold",spaceBefore=4)),
               Spacer(1,0.15*cm),main,Spacer(1,0.25*cm),note])
    return buf.getvalue()

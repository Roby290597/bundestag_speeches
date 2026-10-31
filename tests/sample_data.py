"""Kleines, handgebautes Plenarprotokoll im Format der Bundestag-Open-Data-XMLs.

Liegt als Python-String vor, weil *.xml in der .gitignore steht.
"""

SITZUNG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<dbtplenarprotokoll>
  <vorspann>
    <inhaltsverzeichnis/>
  </vorspann>
  <sitzungsverlauf>
    <tagesordnungspunkt top-id="TOP 1">
      <rede id="ID1">
        <p klasse="redner"><redner id="1"><name><vorname>Anna</vorname><nachname>Beispiel</nachname><fraktion>SPD</fraktion></name></redner>Anna Beispiel (SPD):</p>
        <p klasse="J_1">Sehr geehrte Frau Präsidentin! Wir brauchen mehr Investitionen in Schulen.</p>
        <kommentar>(Beifall bei der SPD)</kommentar>
        <p klasse="O">Die Kommunen dürfen nicht allein gelassen werden.</p>
      </rede>
      <rede id="ID2">
        <p klasse="redner"><redner id="2"><name><vorname>Bernd</vorname><nachname>Muster</nachname><fraktion>CDU/CSU</fraktion></name></redner>Bernd Muster (CDU/CSU):</p>
        <p klasse="J_1">Die Schuldenbremse muss eingehalten werden.</p>
        <kommentar>(Zuruf von der SPD: Unsinn!)</kommentar>
      </rede>
      <rede id="ID3">
        <p klasse="redner"><redner id="1"><name><vorname>Anna</vorname><nachname>Beispiel</nachname><fraktion>SPD</fraktion></name></redner>Anna Beispiel (SPD):</p>
        <p klasse="J_1">Ich möchte noch etwas zur Digitalisierung sagen.</p>
      </rede>
      <rede id="ID4">
        <p klasse="redner"><redner id="3"><name><vorname>Clara</vorname><nachname>Leer</nachname></name></redner>Clara Leer:</p>
        <kommentar>(Heiterkeit)</kommentar>
      </rede>
    </tagesordnungspunkt>
  </sitzungsverlauf>
  <anlagen>
    <rednerliste sitzungsdatum="31.10.2026">
      <redner id="1"><name><vorname>Anna</vorname><nachname>Beispiel</nachname><fraktion>SPD</fraktion></name></redner>
      <redner id="2"><name><vorname>Bernd</vorname><nachname>Muster</nachname><fraktion>CDU/CSU</fraktion></name></redner>
      <redner id="4"><name><vorname>Dora</vorname><nachname>Ministerin</nachname><rolle><rolle_lang>Bundesministerin für Bildung</rolle_lang><rolle_kurz>Bundesministerin</rolle_kurz></rolle></name></redner>
    </rednerliste>
  </anlagen>
</dbtplenarprotokoll>
"""

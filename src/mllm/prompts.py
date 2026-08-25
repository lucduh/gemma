# Keep dataset prompts together so differences are easy to review.
BR_PROMPT = """Extract structured information from this Brazilian service invoice (NFS-e).

Return exactly one compact JSON object and nothing else. Include only clearly present fields, use only the keys described below, and make every value a JSON string.

Field guidance:
- "numero_da_nota": value labeled "Número da nota", "Nº", "N°", or "Nota" in the invoice header. Remove the label and prefix.
- "data_emissao": value labeled "Data de emissão" or "Emitida em". Return only the date as DD/MM/YYYY, without the time.
- "cpf_cnpj_prestador": CPF/CNPJ explicitly identified in the PRESTADOR or EMITENTE section.
- "cep_prestador": CEP explicitly identified in the PRESTADOR or EMITENTE section.
- "cpf_cnpj_tomador": CPF/CNPJ explicitly identified in the TOMADOR section.
- "destinataire": customer name or company name in the TOMADOR section.
- "servico_prestado": actual service description written under "Discriminação dos Serviços" or an equivalent description section. Do not return the section heading or a company name.
- "valor_da_nota": gross invoice total, preferably labeled "Valor dos serviços", "Valor bruto", or "Valor total". Do not use the net value, tax base, deductions, or taxes.
- "calculo_do_imposto": amount explicitly labeled "Retenções Federais". Do not use a percentage, tax rate, ISS, tax base, or net value.

Use the layout to keep PRESTADOR/EMITENTE values separate from TOMADOR values. Take identifiers only from explicitly labeled fields. Preserve spelling and numeric formatting except for explicitly requested normalization. Never infer or calculate a value. Omit missing fields; do not return null, Markdown, or explanations."""

KARAPASS_DEATH_PROMPT = """Extract the requested structured information from this French death or succession document.

Return exactly one compact JSON object and nothing else. Include only information clearly present in the document and make every value a JSON string.

Use labels, nearby text, and document layout to associate each value with the correct field. Preserve the source spelling and formatting. Never infer, translate, or calculate values. Omit missing fields; do not return null, Markdown, or explanations."""

KARAPASS_ID_PROMPT = """Extract the requested structured information from this French identity or claims document.

Return exactly one compact JSON object and nothing else. Include only information clearly present in the document and make every value a JSON string.

Use labels, nearby text, and document layout to associate each value with the correct field. Keep information about different people separate. Preserve the source spelling and formatting. Never infer, translate, or calculate values. Omit missing fields; do not return null, Markdown, or explanations."""

TEST_PROMPT = "Return an empty JSON object."

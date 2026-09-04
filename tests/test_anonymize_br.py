import unittest

from scripts.anonymize_br import extract_fragment, sanitize_fragment


class ExtractFragmentTest(unittest.TestCase):
    def test_removes_markdown_fence(self):
        response = "```html\n<section>Invoice</section>\n```"
        self.assertEqual(extract_fragment(response), "<section>Invoice</section>")

    def test_extracts_body_from_complete_document(self):
        response = (
            "<html><head><title>x</title></head><body><p>Invoice</p></body></html>"
        )
        self.assertEqual(extract_fragment(response), "<p>Invoice</p>")


class SanitizeFragmentTest(unittest.TestCase):
    def test_removes_executable_markup_and_unsafe_attributes(self):
        fragment = (
            '<section onclick="steal()"><script>secret()</script>'
            '<p class="row bad$class" style="display:none">A & B</p></section>'
        )
        result, _ = sanitize_fragment(fragment)

        self.assertEqual(result, '<section><p class="row">A &amp; B</p></section>')

    def test_keeps_safe_inline_layout_styles(self):
        fragment = (
            '<div style="display: grid; grid-template-columns: 1fr 2fr; '
            'border: 1px solid #222; background-image: url(https://example.test/x)">x</div>'
        )

        result, _ = sanitize_fragment(fragment)

        self.assertIn("display: grid", result)
        self.assertIn("grid-template-columns: 1fr 2fr", result)
        self.assertIn("border: 1px solid #222", result)
        self.assertNotIn("url", result)

    def test_redacts_every_tagged_value(self):
        fragment = (
            '<div class="field"><b>CPF:</b> '
            '<span class="sensitive" data-field="cpf_cnpj_tomador">123.456</span></div>'
        )
        result, fields = sanitize_fragment(fragment, redact_all=True)

        self.assertIn("CPF:", result)
        self.assertNotIn("123.456", result)
        self.assertIn("[REDACTED]</span>", result)
        self.assertEqual(fields, {"cpf_cnpj_tomador"})

    def test_can_redact_only_selected_fields(self):
        fragment = (
            '<span data-field="name">Ana</span><span data-field="amount">10,00</span>'
        )
        result, _ = sanitize_fragment(fragment, redact_fields={"name"})

        self.assertNotIn("Ana", result)
        self.assertIn("10,00", result)


if __name__ == "__main__":
    unittest.main()

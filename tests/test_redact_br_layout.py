import base64
import re
import unittest

from PIL import Image

from scripts.redact_br_layout import (
    image_html,
    parse_redactions,
    pixel_box,
    render_images,
)


class ParseRedactionsTest(unittest.TestCase):
    def test_parses_fenced_json_and_normalizes_boxes(self):
        response = """```json
        {"redactions":[
          {"category":"cpf", "box_2d":[300.2, 400.8, 100, 200]},
          {"category":"invalid value", "box_2d":[1, 2, 3, 4]},
          {"category":"empty", "box_2d":[1, 2, 1, 4]}
        ]}
        ```"""

        redactions = parse_redactions(response)

        self.assertEqual(len(redactions), 1)
        self.assertEqual(redactions[0].category, "cpf")
        self.assertEqual(redactions[0].box_2d, (100, 200, 300, 401))

    def test_rejects_response_without_list(self):
        with self.assertRaises(TypeError):
            parse_redactions('{"answer": []}')


class RenderImagesTest(unittest.TestCase):
    def test_converts_normalized_box_and_applies_padding(self):
        self.assertEqual(pixel_box((100, 200, 300, 400), 100, 200, 2), (18, 18, 42, 62))

    def test_redaction_changes_only_boxed_pixels(self):
        image = Image.new("RGB", (10, 10), "white")
        redactions = parse_redactions(
            '{"redactions":[{"category":"name","box_2d":[200,200,800,800]}]}'
        )

        review, redacted = render_images(image, redactions, padding=0)

        self.assertEqual(image.getpixel((5, 5)), (255, 255, 255))
        self.assertEqual(redacted.getpixel((5, 5)), (0, 0, 0))
        self.assertEqual(redacted.getpixel((0, 0)), (255, 255, 255))
        self.assertNotEqual(review.getpixel((2, 2)), image.getpixel((2, 2)))

    def test_html_embeds_only_the_supplied_redacted_png(self):
        image = Image.new("RGB", (2, 2), "black")

        document = image_html(image, "Example")
        encoded = re.search(r"base64,([A-Za-z0-9+/=]+)", document).group(1)

        self.assertTrue(base64.b64decode(encoded).startswith(b"\x89PNG"))
        self.assertNotIn("file://", document)


if __name__ == "__main__":
    unittest.main()

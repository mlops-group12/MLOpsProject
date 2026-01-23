from locust import HttpUser, between, task

from io import BytesIO
from PIL import Image


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def predict(self) -> None:
        """A task that simulates a user visiting a random item URL of the FastAPI app."""
        img = Image.new("L", (64, 64), color="white")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        files = {"data": ("test.png", img_byte_arr, "image/png")}

        self.client.post("/predict/", files=files)

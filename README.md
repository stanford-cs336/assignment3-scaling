# CS336 Spring 2026 Assignment 3: Scaling

For a full description of the assignment, see the assignment handout at
[cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## For students

Install uv

```sh
uv sync
```

Set `A3_API_KEY` to your 8-digit student ID:

```sh
export A3_API_KEY=06123456
```

The hosted training API is available at:

```text
http://hyperturing.stanford.edu:8000
```

Click here for the [docs](http://hyperturing.stanford.edu:8000/docs) and [dashboard](http://hyperturing.stanford.edu:8000/dashboard).

See [`./examples/client_example.ipynb`](./examples/client_example.ipynb) for an
example of submitting and inspecting training runs.

## For non-students

Install dependencies:

```sh
uv sync
```

To download tokenized data:

```sh
uv run modal run scripts/1_download_tokenized_data.py
```

To run training directly:

```sh
uv run cs336_scaling/training/run.py
```

To run the API and dispatcher, set:

```sh
DATABASE_URL_PROD="postgresql://..."
DATABASE_URL_DEV="postgresql://..."
INTERNAL_API_KEY="SOMEKEY"
```

Then run:

```sh
DB_ENV=prod uv run fastapi run &
DB_ENV=prod uv run dispatcher &
```

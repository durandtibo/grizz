# grizz

<p align="center">
    <a href="https://github.com/durandtibo/grizz/actions">
        <img alt="CI" src="https://github.com/durandtibo/grizz/workflows/CI/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/grizz/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/grizz/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/grizz/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/grizz/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/grizz/">
        <img alt="Documentation" src="https://github.com/durandtibo/grizz/workflows/Documentation%20(stable)/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/grizz/">
        <img alt="Documentation" src="https://github.com/durandtibo/grizz/workflows/Documentation%20(unstable)/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/grizz">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/grizz/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/grizz/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/7f2bd443a970c115cd94/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/grizz/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/7f2bd443a970c115cd94/test_coverage" />
    </a>
    <br/>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
    </a>
    <br/>
    <a href="https://pypi.org/project/grizz/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/grizz">
    </a>
    <a href="https://pypi.org/project/grizz/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/grizz.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/grizz">
    </a>
    <br/>
    <a href="https://pepy.tech/project/grizz">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/grizz">
    </a>
    <a href="https://pepy.tech/project/grizz">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/grizz/month">
    </a>
    <br/>
</p>

## Overview

`grizz` is a light library to ingest and transform data
in [polars](https://docs.pola.rs/api/python/stable/reference/index.html) DataFrame.
`grizz` uses an object-oriented strategy, where ingestors and transformers are building blocks that
can be combined together.
`grizz` can be extend to add custom DataFrame ingestors and transformers.
For example, the following example shows how to change the casting of some columns.

```pycon

>>> import polars as pl
>>> from grizz.transformer import InplaceCast
>>> transformer = InplaceCast(columns=["col1", "col3"], dtype=pl.Int32)
>>> frame = pl.DataFrame(
...     {
...         "col1": [1, 2, 3, 4, 5],
...         "col2": ["1", "2", "3", "4", "5"],
...         "col3": ["1", "2", "3", "4", "5"],
...         "col4": ["a", "b", "c", "d", "e"],
...     }
... )
>>> out = transformer.transform(frame)
>>> out
shape: (5, 4)
┌──────┬──────┬──────┬──────┐
│ col1 ┆ col2 ┆ col3 ┆ col4 │
│ ---  ┆ ---  ┆ ---  ┆ ---  │
│ i32  ┆ str  ┆ i32  ┆ str  │
╞══════╪══════╪══════╪══════╡
│ 1    ┆ 1    ┆ 1    ┆ a    │
│ 2    ┆ 2    ┆ 2    ┆ b    │
│ 3    ┆ 3    ┆ 3    ┆ c    │
│ 4    ┆ 4    ┆ 4    ┆ d    │
│ 5    ┆ 5    ┆ 5    ┆ e    │
└──────┴──────┴──────┴──────┘

```

- [Documentation](https://durandtibo.github.io/grizz/)
- [Installation](#installation)
- [Contributing](#contributing)
- [API stability](#api-stability)
- [License](#license)

## Documentation

- [latest (stable)](https://durandtibo.github.io/grizz/): documentation from the latest stable
  release.
- [main (unstable)](https://durandtibo.github.io/grizz/main/): documentation associated to the
  main branch of the repo. This documentation may contain a lot of work-in-progress/outdated/missing
  parts.

## Installation

We highly recommend installing
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
`grizz` can be installed from pip using the following command:

```shell
pip install grizz
```

To make the package as slim as possible, only the minimal packages required to use `grizz` are
installed.
To include all the dependencies, you can use the following command:

```shell
pip install grizz[all]
```

Please check the [get started page](https://durandtibo.github.io/grizz/get_started) to see how to
install only some specific dependencies or other alternatives to install the library.
The following is the corresponding `grizz` versions and their dependencies.

| `grizz` | `coola`        | `iden`         | `objectory`  | `polars`     | `python`      |
|---------|----------------|----------------|--------------|--------------|---------------|
| `main`  | `>=0.8.6,<1.0` | `>=0.2,<1.0`   | `>=0.2,<1.0` | `>=1.0,<2.0` | `>=3.9,<3.14` |
| `0.1.1` | `>=0.8.5,<1.0` | `>=0.1,<1.0`   | `>=0.2,<1.0` | `>=1.0,<2.0` | `>=3.9,<3.14` |
| `0.1.0` | `>=0.8.4,<1.0` | `>=0.1.0,<1.0` | `>=0.2,<1.0` | `>=1.0,<2.0` | `>=3.9,<3.14` |
| `0.0.5` | `>=0.7,<1.0`   | `>=0.0.4,<1.0` | `>=0.1,<1.0` | `>=1.0,<2.0` | `>=3.9,<3.13` |
| `0.0.4` | `>=0.7,<1.0`   | `>=0.0.4,<1.0` | `>=0.1,<1.0` | `>=1.0,<2.0` | `>=3.9,<3.13` |

Optional dependencies

| `grizz` | `clickhouse-connect`<sup>*</sup> | `colorlog`<sup>*</sup> | `numpy`<sup>*</sup> | `pyarrow`<sup>*</sup> | `scikit-learn`<sup>*</sup> | `tqdm`<sup>*</sup> |
|---------|----------------------------------|------------------------|---------------------|-----------------------|----------------------------|--------------------|
| `main`  | `>=0.7,<1.0`                     | `>=6.7,<7.0`           | `>=1.22,<3.0`       | `>=10.0,<20.0`        | `>=1.3,<2.0`               | `>=4.65,<5.0`      |
| `0.1.1` | `>=0.7,<1.0`                     |                        |                     | `>=10.0,<19.0`        |                            | `>=4.65,<5.0`      |
| `0.1.0` | `>=0.7,<1.0`                     |                        |                     | `>=10.0,<18.0`        |                            | `>=4.65,<5.0`      |
| `0.0.5` | `>=0.7,<1.0`                     |                        |                     | `>=10.0,<18.0`        |                            | `>=4.65,<5.0`      |
| `0.0.4` | `>=0.7,<1.0`                     |                        |                     | `>=10.0,<17.0`        |                            | `>=4.65,<5.0`      |

<sup>*</sup> indicates an optional dependency

## Contributing

Please check the instructions in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Suggestions and Communication

Everyone is welcome to contribute to the community.
If you have any questions or suggestions, you can
submit [Github Issues](https://github.com/durandtibo/grizz/issues).
We will reply to you as soon as possible. Thank you very much.

## API stability

:warning: While `grizz` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `grizz` to a new version will possibly break any code that
was using the old version of `grizz`.

## License

`grizz` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.

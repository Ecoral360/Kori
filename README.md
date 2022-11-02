# Kori: the automatic python code corrector!

### Future features:

* [ ] StrTest type (currently `str|re.Pattern|list[str|re.Pattern]`) into a class to provide more functionnality (
  See `StrTest` below)
* [ ] KoriTest can take a `() => list[KoriTestAction]`
* [ ] KoriParamTest
* [ ] Conditional actions

#### StrTest

* [ ] Can take a `str`, a `re.Pattern` or a `list[str|re.Pattern]` as an argument
* [ ] `ignore: str | re.Pattern` field
* [ ] Can take a StrTestConfig

#### KoriParamTest
  
from CommonTools.dtyper import _BaseDtype, _DtypeBase, \
    Array, Bool, DateT, DateTimeT, Decimal, Float, Integer, Map, String, \
    check_auto_cast_precision


class DtypeClickHouse(_DtypeBase):
    @classmethod
    def nullable_wrapper(cls, d: _BaseDtype, s: str) -> str:
        if d.nullable:
            s = f"Nullable({s})"
        return s

    @classmethod
    def toInteger(cls, d: Integer) -> str:
        precision = check_auto_cast_precision(
            d.precision,
            supported_precision=[8, 16, 32, 64, 128, 256]
        )
        base = f"Int{precision}"
        if not d.signed:
            base = "U" + base
        return cls.nullable_wrapper(d, base)

    @classmethod
    def toFloat(cls, d: Float) -> str:
        precision = check_auto_cast_precision(
            d.precision,
            supported_precision=[32, 64]
        )
        base = f"Float{precision}"
        return cls.nullable_wrapper(d, base)

    @classmethod
    def toDecimal(cls, d: Decimal) -> str:
        base = f"Decimal({d.precision},{d.scale})"
        return cls.nullable_wrapper(d, base)

    @classmethod
    def toBool(cls, d: Bool) -> str:
        return cls.nullable_wrapper(d, s = "Bool")

    @classmethod
    def toString(cls, d: String) -> str:
        return cls.nullable_wrapper(d, s = "String")

    @classmethod
    def toDateT(cls, d: DateT) -> str:
        return cls.nullable_wrapper(d, s = "Date32")

    @classmethod
    def toDateTimeT(cls, d: DateTimeT) -> str:
        if d.tz is not None:
            base = f"DateTime64({d.precision},'{d.tz}')"
        else:
            base = f"DateTime64({d.precision})"
        return cls.nullable_wrapper(d, s = base)

    @classmethod
    def toArray(cls, d: Array) -> str:
        e = d.element_dtype
        e.nullable = False  # TODO:  Array does not support nullable
        cls_name = e.__class__.__name__
        func = getattr(DtypeClickHouse, f"to{cls_name}")
        base = func(e)
        return f"Array({base})"

    @classmethod
    def toMap(cls, d: Map) -> str:
        k = d.key_dtype
        v = d.value_dtype
        k.nullable = False   # TODO:  key does not support nullable
        k_cls = k.__class__.__name__
        v_cls = v.__class__.__name__
        func_k = getattr(DtypeClickHouse, f"to{k_cls}")
        func_v = getattr(DtypeClickHouse, f"to{v_cls}")
        base_k = func_k(k)
        base_v = func_v(v)
        return f"Map({base_k},{base_v})"
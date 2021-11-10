from pyspark.sql import SparkSession
from pyspark.sql.types import DataType, NullType, AtomicType, IntegralType, NumericType, FractionalType, \
        StringType, BinaryType, BooleanType, DateType, TimestampType, DecimalType, DoubleType, \
        FloatType, ByteType, IntegerType, LongType, ShortType, ArrayType, MapType, StructField, \
        StructType

def sql_types_example(spark):
    
    # DataType
    dp = DataType()
    
    python_obj = dp.fromInternal(1)
    print(python_obj, type(python_obj))
    
    sql_obj = dp.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(dp.json())
    print(dp.jsonValue())
    print(dp.needConversion())
    print(dp.simpleString())
    print(DataType.typeName())

    # NullType
    nt = NullType()
    
    python_obj = nt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = nt.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(nt.json())
    print(nt.jsonValue())
    print(nt.needConversion())
    print(nt.simpleString())
    print(NullType.typeName())

    # AtomicType
    at = AtomicType()
    
    python_obj = at.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = at.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(at.json())
    print(at.jsonValue())
    print(at.needConversion())
    print(at.simpleString())
    print(AtomicType.typeName())

    # NumericType
    nt = NumericType()
    
    python_obj = nt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = nt.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(nt.json())
    print(nt.jsonValue())
    print(nt.needConversion())
    print(nt.simpleString())
    print(NumericType.typeName())

    # IntegralType
    it = IntegralType()
    
    python_obj = it.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = it.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(it.json())
    print(it.jsonValue())
    print(it.needConversion())
    print(it.simpleString())
    print(IntegralType.typeName())

    # FractionalType
    ft = FractionalType()
    
    python_obj = ft.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = ft.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(ft.json())
    print(ft.jsonValue())
    print(ft.needConversion())
    print(ft.simpleString())
    print(FractionalType.typeName())

    # StringType
    st = StringType()
    
    python_obj = st.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = st.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(st.json())
    print(st.jsonValue())
    print(st.needConversion())
    print(st.simpleString())
    print(StringType.typeName())

    # BinaryType
    bt = BinaryType()
    
    python_obj = bt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = bt.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(bt.json())
    print(bt.jsonValue())
    print(bt.needConversion())
    print(bt.simpleString())
    print(BinaryType.typeName())

    # BooleanType
    bt = BooleanType()
    
    python_obj = bt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = bt.toInternal(1)
    print(sql_obj, type(sql_obj))
    
    print(bt.json())
    print(bt.jsonValue())
    print(bt.needConversion())
    print(bt.simpleString())
    print(BooleanType.typeName())

    # DateType
    from datetime import datetime
    dt = DateType()
    
    python_obj = dt.fromInternal(1000)
    print(python_obj, type(python_obj))
    
    today = datetime.today()
    sql_obj = dt.toInternal(today)
    print(sql_obj, type(sql_obj))

    print(dt.json())
    print(dt.jsonValue())
    print(dt.needConversion())
    print(dt.simpleString())
    print(DateType.typeName())

    # TimestampType
    tt = TimestampType()
    
    python_obj = tt.fromInternal(365000000)
    print(python_obj, type(python_obj))

    today = datetime.today()
    sql_obj = tt.toInternal(today)
    print(sql_obj, type(sql_obj))
    
    print(tt.json())
    print(tt.jsonValue())
    print(tt.needConversion())
    print(tt.simpleString())
    print(TimestampType.typeName())

    # DecimalType
    dt = DecimalType()

    python_obj = dt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = dt.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(dt.json())
    print(dt.jsonValue())
    print(dt.needConversion())
    print(dt.simpleString())
    print(DecimalType.typeName())

    # DoubleType
    dt = DoubleType()

    python_obj = dt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = dt.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(dt.json())
    print(dt.jsonValue())
    print(dt.needConversion())
    print(dt.simpleString())
    print(DoubleType.typeName())

    # FloatType
    ft = FloatType()

    python_obj = ft.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = ft.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(ft.json())
    print(ft.jsonValue())
    print(ft.needConversion())
    print(ft.simpleString())
    print(FloatType.typeName())

    # ByteType
    bt = ByteType()

    python_obj = bt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = bt.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(bt.json())
    print(bt.jsonValue())
    print(bt.needConversion())
    print(bt.simpleString())
    print(ByteType.typeName())

    # IntegerType
    it = IntegerType()
    
    python_obj = it.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = it.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(it.json())
    print(it.jsonValue())
    print(it.needConversion())
    print(it.simpleString())
    print(IntegerType.typeName())

    # LongType
    lt = LongType()

    python_obj = lt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = lt.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(lt.json())
    print(lt.jsonValue())
    print(lt.needConversion())
    print(lt.simpleString())
    print(LongType.typeName())

    # ShortType
    st = ShortType()

    python_obj = st.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = st.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(st.json())
    print(st.jsonValue())
    print(st.needConversion())
    print(st.simpleString())
    print(ShortType.typeName())

    # ArrayType
    dt = DataType()
    at = ArrayType(dt)

    python_obj = at.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = at.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(at.json())
    print(at.jsonValue())
    print(at.needConversion())
    print(at.simpleString())
    print(ArrayType.typeName())
    print(ArrayType.fromJson({"containsNull": True, "elementType": "string"}))

    # MapType
    key_type = DataType()
    value_type = DataType()
    mt = MapType(key_type, value_type)
    
    python_obj = mt.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = mt.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(mt.json())
    print(mt.jsonValue())
    print(mt.needConversion())
    print(mt.simpleString())
    print(MapType.typeName())
    print(MapType.fromJson({"valueContainsNull": True, "keyType": "string", "valueType": "integer"}))

    # StructField
    dt = DataType()
    sf = StructField("first_struct", dt)

    python_obj = sf.fromInternal(1)
    print(python_obj, type(python_obj))

    sql_obj = sf.toInternal(1)
    print(sql_obj, type(sql_obj))

    print(sf.json())
    print(sf.jsonValue())
    print(sf.needConversion())
    print(sf.simpleString())
    print(StructField.fromJson({"metadata": None, "nullable": True, "name": "first_struct", "type": "string"}))

    # StructType
    string_type = StringType()
    st = StructType([StructField("first_struct", StringType()), StructField("second_struct", DataType())])
    print("------")
    print(st.names)
    print(st.fields)
    print(st._needConversion)
    print(st._needSerializeAnyField)

    python_obj = st.fromInternal(["first_struct", "second_struct"])
    print(python_obj, type(python_obj))

    sql_obj = st.toInternal(["first_struct", "second_struct"])
    print(sql_obj, type(sql_obj))

    print(st.json())
    print(st.jsonValue())
    print(st.needConversion())
    print(st.simpleString())
    print(st.fieldNames())
    fields = {
        "fields": [
            {"metadata": None, "nullable": True, "name": "first", "type": "string"},
            {"metadata": None, "nullable": True, "name": "second", "type": "integer"}
        ]
    }
    print(st.fromJson(fields))
    
    st.add(StructField("first_struct", StringType()))
    
    print("st.add success!")


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Types example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_types_example(spark)

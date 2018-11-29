function reducer(_, values) {
  var idsDict = {};

  function red(acc, item) {
    acc.purchase += item.purchase;
    acc.count += item.count;
    if (item.id) {
      idsDict[item.id] = true;
    } else {
      idsDict = Object.assign(idsDict, item.idsDict);
    }
    return acc;
  }
  var result = values.reduce(red, { count: 0, purchase: 0 });
  result.idsDict = idsDict;
  return result;
}

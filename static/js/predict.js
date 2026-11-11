(function () {
  const presets = {
    normal: { amount: 45, avg_amount_30d: 50, merchant_category: 'grocery', hour: 14, day_of_week: 2,
      distance_from_home_km: 4, distance_from_last_txn_km: 2, txn_count_24h: 2, card_present: 1, international: 0 },
    fraud: { amount: 1850, avg_amount_30d: 60, merchant_category: 'electronics', hour: 3, day_of_week: 5,
      distance_from_home_km: 140, distance_from_last_txn_km: 95, txn_count_24h: 9, card_present: 0, international: 1 }
  };
  document.querySelectorAll('.chip').forEach(b => b.addEventListener('click', () => {
    const p = presets[b.dataset.p], f = document.getElementById('pf');
    Object.entries(p).forEach(([k, v]) => { f.elements[k].value = v; });
  }));
})();

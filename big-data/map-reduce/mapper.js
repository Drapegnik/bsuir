function mapper() {
  emit(this.Gender, {
    purchase: this.Purchase,
    id: this.User_ID,
    count: 1
  });
}

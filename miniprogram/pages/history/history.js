Page({
  data: {
    list: [],
    expanded: -1
  },

  onShow() {
    this.setData({ list: wx.getStorageSync('ask_history') || [] })
  },

  toggle(e) {
    const idx = e.currentTarget.dataset.idx
    this.setData({ expanded: this.data.expanded === idx ? -1 : idx })
  },

  clearAll() {
    wx.showModal({
      title: '清空历史记录',
      content: '确定删除全部历史问诊记录吗？',
      success: (res) => {
        if (res.confirm) {
          wx.removeStorageSync('ask_history')
          this.setData({ list: [], expanded: -1 })
        }
      }
    })
  }
})
